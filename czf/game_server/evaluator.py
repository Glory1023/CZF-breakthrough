'''CZF Game Server'''
import asyncio
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import ctypes
import multiprocessing as mp
import platform
import time
import typing
import sys
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from czf.env import czf_env, atari_env
from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer, Queue


async def run_eval_env_manager(*args):
    '''run :class:`EvalEnvManager`'''
    manager = EvalEnvManager(*args)
    await manager.loop()


class ModelInfo(ctypes.Structure):
    '''ModelInfo shared between processes'''
    _fields_ = [('name', ctypes.c_wchar_p), ('version', ctypes.c_int)]


@dataclass
class EnvInfo:
    '''Environment information

    :param state: the game simulator state
    :param trajectory: a segment of trajectory (not necessary from initial)
    :param workers: all workers that has handled the job (reserved for affinity)
    '''
    state: typing.Any
    trajectory: czf_pb2.Trajectory
    workers: list


class EvalEnvManager:
    '''Game Environment Manager'''
    def __init__(self, args, config, callbacks, proc_index, model_info_1p, model_info_2p,
                 pipe, job_queue, result_queue):
        self._algorithm = config['algorithm']
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}',
                                  hostname=platform.node())
        # multiprocess
        self._proc_index = proc_index
        self._models = {
            czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P: model_info_1p,
            czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P: model_info_2p,
            czf_pb2.Job.Operation.MUZERO_EVALUATE_1P: model_info_1p,
            czf_pb2.Job.Operation.MUZERO_EVALUATE_2P: model_info_2p,
        }
        self._pipe = pipe
        self._job_queue = job_queue
        self._result_queue = result_queue
        # callbacks
        self._action_policy_fn = callbacks['action_policy']
        metric_callbacks = callbacks.get('metric', {})
        self._after_apply_callback = metric_callbacks.get('after_apply', None)
        # tree config
        mcts_config = config['mcts']
        if self._algorithm == 'AlphaZero':
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=0.,
            )
        elif self._algorithm == 'MuZero':
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                tree_min_value=mcts_config.get('tree_min_value', float('inf')),
                tree_max_value=mcts_config.get('tree_max_value', float('-inf')),
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=0.,
                discount=mcts_config.get('discount', 1.),
            )
            self._mstep = max(mcts_config['nstep'],
                          config['learner']['rollout_steps'])
        # game_server config
        self._sequence = config['game_server']['sequence']
        # game env
        self._num_env = args.num_env * 2
        game_config = config['game']
        obs_config = game_config['observation']
        env_name = game_config['name']
        self._video_dir = None
        if env_name in czf_env.available_games():
            self._game = czf_env.load_game(env_name)
            # check game config in lightweight game
            assert self._game.num_players == game_config['num_player']
            assert self._game.num_distinct_actions == game_config['actions']
            assert self._game.observation_tensor_shape == [
                obs_config['channel'], *obs_config['spatial_shape']
            ]
        else:
            self._game = atari_env.load_game(env_name,
                                             obs_config['frame_stack'])
            self._video_dir = 'demo/videos/' + args.suffix + '_' + datetime.today(
            ).strftime('%Y%m%d_%H%M')
        self._frame_stack = obs_config['frame_stack']
        self._envs = [None] * self._num_env
        self._total_rewards = [None] * self._num_env
        self._num_steps = [None] * self._num_env
        self._operations = [None] * self._num_env
        if self._algorithm == 'AlphaZero':
            for i in range(args.num_env):
                self._operations[i] = [
                    czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P,
                    czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P,
                ]
                self._operations[args.num_env + i] = [
                    czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P,
                    czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P,
                ]
        elif self._algorithm == 'MuZero':
            for i in range(args.num_env):
                self._operations[i] = [
                    czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                    czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
                ]
                self._operations[args.num_env + i] = [
                    czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
                    czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                ]
        for index in range(self._num_env):
            self.__reset(index, new=True)

    async def loop(self):
        '''main loop'''
        while True:
            raw = self._pipe.recv()
            job = czf_pb2.Job.FromString(raw)
            # print('loop')
            # print(job)
            if not job.HasField('payload'):
                # if flush model has done, start jobs for each env
                operation = job.procedure[0]
                for index in range(self._num_env):
                    if self._operations[index][0] == operation:
                        self.__send_search_job(index)
                continue
            self.__on_job_completed(job)

    def __reset(self, env_index, new=False):
        '''reset env of index'''
        if not new:
            state = self._envs[env_index].state
        if not new and hasattr(state, 'reset'):
            state.reset()
        else:
            if self._video_dir is not None and self._proc_index == 0 and env_index == 0:
                state = self._game.new_initial_state(video_dir=self._video_dir)
            else:
                state = self._game.new_initial_state()
        self._envs[env_index] = EnvInfo(
            state,
            czf_pb2.Trajectory(),
            [czf_pb2.Node()],
        )
        self._total_rewards[env_index] = [0.] * self._game.num_players
        self._num_steps[env_index] = 0

    def __send_search_job(self, env_index):
        '''helper to send a `Job` to actor'''
        env = self._envs[env_index]
        current_player = self._envs[env_index].state.current_player
        operation = self._operations[env_index][current_player]
        model_info = self._models[operation]
        # workers = [czf_pb2.Node(identity='g', hostname=str(time.time()))] * 2
        if self._algorithm == 'AlphaZero':
            state=czf_pb2.WorkerState(
                serialized_state=env.state.serialize(),
            )
        elif self._algorithm == 'MuZero':
            state=czf_pb2.WorkerState(
                legal_actions=env.state.legal_actions,
                observation_tensor=env.state.observation_tensor,
            )
        job = czf_pb2.Job(
            model=czf_pb2.ModelInfo(name=model_info.name,
                                    version=model_info.version),
            procedure=[operation],
            step=0,
            payload=czf_pb2.Job.Payload(
                env_index=self._proc_index * self._num_env + env_index,
            )
        )
        job.initiator.CopyFrom(self._node)
        # job.payload.state.workers.CopyFrom(env.workers)
        job.payload.state.CopyFrom(state)
        job.payload.state.tree_option.CopyFrom(self._tree_option)
        packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job]))
        # print(packet)
        self._job_queue.put(packet.SerializeToString())
        # self.start_time[env_index] = time.time()

    def __on_job_completed(self, job: czf_pb2.Job):
        '''callback on job completion'''
        env_index = job.payload.env_index % self._num_env
        env = self._envs[env_index]
        # env.workers[:] = job.workers
        # choose action according to the policy
        evaluated_state = job.payload.state
        policy = evaluated_state.evaluation.policy
        legal_actions = env.state.legal_actions
        legal_actions_policy = [policy[action] for action in legal_actions]
        num_moves = self._num_steps[env_index]
        chosen_action = self._action_policy_fn(num_moves, legal_actions,
                                               legal_actions_policy)
        # apply action
        # print(job.workers, time.time())
        # print('apply', self._proc_index, env_index,
        #       time.time() - self.start_time[env_index])
        env.state.apply_action(chosen_action)
        self._num_steps[env_index] += 1
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state, env.state)
        # game transition
        for player, reward in enumerate(env.state.rewards):
            self._total_rewards[env_index][player] += reward

        if env.state.is_terminal:
            first_player = self._operations[env_index][0]
            if first_player == czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P:
                reward = env.state.rewards[0]
            else:  # first_player == czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P:
                reward = env.state.rewards[1]
            if first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_1P:
                reward = env.state.rewards[0]
            else:  # first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_2P:
                reward = env.state.rewards[1]
            self._result_queue.put(reward)
            self.__reset(env_index)
        else:
            # send a search job
            self.__send_search_job(env_index)

class EvalGameServer:
    '''Evalutation Game Server'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}',
                                  hostname=platform.node())
        # server mode
        algorithm = config['algorithm']
        assert algorithm in ('AlphaZero', 'MuZero')
        print(f'[{algorithm} Evaluation Mode]', self._node.identity)
        # game config
        game_config = config['game']
        self._num_player = game_config['num_player']
        # evaluation config
        eval_config = config['evaluator']
        self._first_ckpt = eval_config['first_checkpoint']
        self._last_ckpt = eval_config['last_checkpoint']
        self._freq = eval_config['frequency']
        self._best_elo = eval_config['elo_base']
        self._replace_rate = eval_config['replace_rate']
        # model
        self._model_info_1p = mp.Value(ModelInfo, 'default', self._first_ckpt)
        self._model_info_2p = mp.Value(ModelInfo, 'default', self._first_ckpt)
        # tensorboard log
        storage_path = Path(args.storage_dir)
        log_path = storage_path / 'log' / 'eval'
        if not log_path.exists():
            log_path.mkdir(exist_ok=True)
        self._summary_writer = SummaryWriter(log_dir=log_path,
                                             purge_step=self._first_ckpt)
        # start EvalEnvManager
        self._num_env = args.num_env * self._num_player
        self._total = self._num_env * args.num_proc
        self._job_queue = Queue()
        self._result_queue = Queue()
        self._pipe = [mp.Pipe() for i in range(args.num_proc)]
        self._manager = [
            mp.Process(
                target=lambda *args: asyncio.run(run_eval_env_manager(*args)),
                args=(
                    args,
                    config,
                    callbacks,
                    index,
                    self._model_info_1p,
                    self._model_info_2p,
                    pipe,
                    self._job_queue,
                    self._result_queue,
                )) for index, (_, pipe) in enumerate(self._pipe)
        ]
        for manager in self._manager:
            manager.start()

        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )
        asyncio.create_task(self.__send_load_model())

    def terminate(self):
        '''terminate all EvalEnvManager'''
        for manager in self._manager:
            manager.join()

    async def loop(self):
        '''main loop'''
        await asyncio.gather(
            self._recv_job_loop(),
            self._send_job_loop(),
            self._write_result_loop(),
        )

    async def _recv_job_loop(self):
        '''a loop to receive `Job`'''
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'job':
                job = packet.job
                if not job.HasField('payload'):
                    # if flush model has done, start jobs for each env
                    # print(job)
                    for pipe in self._pipe:
                        pipe[0].send(job.SerializeToString())
                    continue
                index = job.payload.env_index // self._num_env
                self._pipe[index][0].send(job.SerializeToString())
            elif packet_type == 'job_batch':
                jobs = packet.job_batch.jobs
                for job in jobs:
                    if not job.HasField('payload'):
                        # if flush model has done, start jobs for each env
                        for pipe in self._pipe:
                            pipe[0].send(job.SerializeToString())
                        continue
                    index = job.payload.env_index // self._num_env
                    self._pipe[index][0].send(job.SerializeToString())

    async def _send_job_loop(self):
        '''a loop to send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._job_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            # print('send job')
            await self._broker.send(raw)

    async def _write_result_loop(self):
        '''a loop to write evaluation result'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._result_queue
        eval_result = []
        while True:
            result = await loop.run_in_executor(executor, queue.get)
            eval_result.append(result)
            if len(eval_result) == self._total:
                asyncio.create_task(self.__write_result(eval_result))
                eval_result = []

    async def __write_result(self, eval_result: list):
        '''helper to process evaluation result'''
        total = len(eval_result)
        win_rate = eval_result.count(1) / total
        draw_rate = eval_result.count(0) / total
        lose_rate = eval_result.count(-1) / total
        # score for win: 2, draw: 1, lose: 0
        score = np.sum(np.array(eval_result) + 1) / (2 * total)
        elo_1p_diff = 400 * np.log10(score / (1 - score))
        elo_1p = elo_1p_diff + self._best_elo
        print(
            '{}-{} Win: {:.2%} Draw: {:.2%} Lose: {:.2%} Current Elo: {:.1f} ({:+.1f})'
            .format(
                self._model_info_1p.version,
                self._model_info_2p.version,
                win_rate,
                draw_rate,
                lose_rate,
                elo_1p,
                elo_1p_diff,
            ))
        # send the evaluation result
        # result = czf_pb2.EvaluationResult(
        #     iteration=self._model_info_1p.version,
        #     elo=elo_1p,
        #     win=win_rate,
        #     draw=draw_rate,
        #     lose=lose_rate,
        # )
        # result.target.CopyFrom(self._model_info_1p)
        # result.base.CopyFrom(self._model_info_2p)
        # packet = czf_pb2.Packet(evaluation_result=result)
        # await self._upstream.send(packet.SerializeToString())

        # write the evaluation result to tensorboard
        step = self._model_info_1p.version
        writer = self._summary_writer
        writer.add_scalar('eval/elo', elo_1p, step)
        writer.add_scalar('eval/current_version', self._model_info_1p.version, step)
        writer.add_scalar('eval/best_version', self._model_info_2p.version, step)
        writer.add_scalars('eval/result', {
            'win': win_rate,
            'draw': draw_rate,
            'lose': lose_rate,
        }, step)
        writer.flush()

        # update model according to the score
        if score > self._replace_rate:  # current model (1p) is the best
            # self._model_info_2p.name = self._model_info_1p.name
            self._model_info_2p.version = self._model_info_1p.version
            self._best_elo = elo_1p

        if self._model_info_1p.version >= self._last_ckpt:
            print('Done')
            return
        await self.__send_load_model()

    async def __send_load_model(self):
        '''helper to wait for new model and send jobs to flush'''
        next_version = self._model_info_1p.version + self._freq
        self._model_info_1p.version = next_version
        # send jobs to flush model
        job1 = czf_pb2.Job(
            model=czf_pb2.ModelInfo(name=self._model_info_1p.name,
                                    version=self._model_info_1p.version),
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_1P],
            step=0,
        )
        job1.initiator.CopyFrom(self._node)
        packet1 = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job1]))
        # print(packet1)
        await self._broker.send(packet1.SerializeToString())
        job2 = czf_pb2.Job(
            model=czf_pb2.ModelInfo(name=self._model_info_2p.name,
                                    version=self._model_info_2p.version),
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_2P],
            step=0,
        )
        job2.initiator.CopyFrom(self._node)
        packet2 = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job2]))
        # print(packet2)
        await self._broker.send(packet2.SerializeToString())
