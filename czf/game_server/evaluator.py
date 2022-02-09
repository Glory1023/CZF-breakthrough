'''CZF Game Server'''
import asyncio
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from czf.env import atari_env, czf_env
from czf.game_server.game_server import EnvInfo, ModelInfo
from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer, Queue

EvalResult = namedtuple('EvalResult', [
    'game_steps',
    'total_rewards',
])


async def run_eval_env_manager(*args):
    '''run :class:`EvalEnvManager`'''
    manager = EvalEnvManager(*args)
    await manager.loop()


class EvalEnvManager:
    '''Game Environment Manager'''
    def __init__(self, args, config, callbacks, proc_index, pipe, job_queue, result_queue,
                 operation, model_info):
        self._algorithm = config['algorithm']
        self._num_players = config['game']['num_player']
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}', hostname=platform.node())
        # multiprocess
        self._proc_index = proc_index
        self._pipe = pipe
        self._job_queue = job_queue
        self._result_queue = result_queue
        self._operation = operation
        self._model_info = model_info
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
            self._mstep = max(mcts_config['nstep'], config['learner']['rollout_steps'])
        # game env
        self._num_env = args.num_env * self._num_players
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
            self._game = atari_env.load_game(env_name, obs_config['frame_stack'])
            self._video_dir = 'demo/videos/' + args.suffix + '_' + datetime.today().strftime(
                '%Y%m%d_%H%M')
        self._frame_stack = obs_config['frame_stack']
        self._envs = [None] * self._num_env
        self._total_rewards = [None] * self._num_env
        self._num_steps = [None] * self._num_env
        self._player_roles = [None] * self._num_env
        # player role
        if self._num_players == 1:
            for i in range(args.num_env):
                self._player_roles[i] = ['1P']
        elif self._num_players == 2:
            for i in range(args.num_env):
                self._player_roles[i] = ['1P', '2P']
                self._player_roles[args.num_env + i] = ['2P', '1P']
        # reset all environments
        for index in range(self._num_env):
            self.__reset(index, new=True)

    async def loop(self):
        '''main loop'''
        while True:
            raw = self._pipe.recv()
            job = czf_pb2.Job.FromString(raw)
            # print(job)
            if not job.HasField('payload'):
                # if flush model has done, start jobs for each env
                operation = job.procedure[0]
                for index in range(self._num_env):
                    player_role = self._player_roles[index][0]
                    if self._operation[player_role] == operation:
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
        # print(f'__send_search_job: process {self._proc_index} env {env_index}')
        env = self._envs[env_index]
        current_player = self._envs[env_index].state.current_player
        player_role = self._player_roles[env_index][current_player]
        operation = self._operation[player_role]
        model_info = self._model_info[player_role]

        if self._algorithm == 'AlphaZero':
            state = czf_pb2.WorkerState(serialized_state=env.state.serialize(), )
        elif self._algorithm == 'MuZero':
            state = czf_pb2.WorkerState(
                legal_actions=env.state.legal_actions,
                observation_tensor=env.state.observation_tensor,
            )
        job = czf_pb2.Job(
            model=czf_pb2.ModelInfo(name=model_info.name, version=model_info.version),
            procedure=[operation],
            step=0,
            payload=czf_pb2.Job.Payload(env_index=self._proc_index * self._num_env + env_index, ))
        job.initiator.CopyFrom(self._node)
        job.payload.state.CopyFrom(state)
        job.payload.state.tree_option.CopyFrom(self._tree_option)
        packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job]))
        # print(packet)
        self._job_queue.put(packet.SerializeToString())

    def __on_job_completed(self, job: czf_pb2.Job):
        '''callback on job completion'''
        env_index = job.payload.env_index % self._num_env
        # print(f'__on_job_completed: process {self._proc_index} env {env_index}')
        env = self._envs[env_index]
        # choose action according to the policy
        evaluated_state = job.payload.state
        policy = evaluated_state.evaluation.policy
        legal_actions = env.state.legal_actions
        legal_actions_policy = [policy[action] for action in legal_actions]
        num_moves = self._num_steps[env_index]
        chosen_action = self._action_policy_fn(num_moves, legal_actions, legal_actions_policy)
        # apply action
        env.state.apply_action(chosen_action)
        # print(chosen_action)
        self._num_steps[env_index] += 1
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state, env.state)
        # game transition
        for player, reward in enumerate(env.state.rewards):
            self._total_rewards[env_index][player] += reward

        if env.state.is_terminal:
            game_steps = self._num_steps[env_index]
            first_player_role = self._player_roles[env_index][0]
            if first_player_role == '1P':
                total_rewards = self._total_rewards[env_index][0]
            elif first_player_role == '2P':
                total_rewards = self._total_rewards[env_index][1]
            result = EvalResult(game_steps=game_steps, total_rewards=total_rewards)
            self._result_queue.put(result)
            self.__reset(env_index)
        else:
            self.__send_search_job(env_index)


class EvalGameServer:
    '''Evalutation Game Server'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(
            identity=f'game-server-{args.suffix}',
            hostname=platform.node(),
        )
        # server mode
        algorithm = config['algorithm']
        assert algorithm in ('AlphaZero', 'MuZero')
        print(f'[{algorithm} Evaluation Mode]', self._node.identity)
        # game config
        game_config = config['game']
        self._num_players = game_config['num_player']
        # evaluation config
        eval_config = config['evaluator']
        self._first_ckpt = eval_config.get('first_checkpoint', 0)
        self._last_ckpt = eval_config.get('last_checkpoint', self._first_ckpt)
        self._freq = eval_config.get('frequency', 1)
        self._latest = eval_config.get('latest', False)
        assert self._last_ckpt >= self._first_ckpt
        assert self._freq > 0

        # a queue to store model version to evaluate
        self._model_version_queue = Queue()
        if self._latest:
            self._last_ckpt = self._first_ckpt
            self._model_version_queue.put(self._last_ckpt)
        else:
            for version in range(self._first_ckpt, self._last_ckpt + 1, self._freq):
                self._model_version_queue.put(version)

        # for two-player game
        self._best_elo = eval_config.get('elo_base', 0.)
        self._replace_rate = eval_config.get('replace_rate', 0.)
        # operation
        if algorithm == 'AlphaZero':
            if self._num_players == 1:
                self._operation = {
                    '1P': czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P,
                }
            elif self._num_players == 2:
                self._operation = {
                    '1P': czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P,
                    '2P': czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P,
                }
        elif algorithm == 'MuZero':
            if self._num_players == 1:
                self._operation = {
                    '1P': czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                }
            elif self._num_players == 2:
                self._operation = {
                    '1P': czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                    '2P': czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
                }
        # model
        if self._num_players == 1:
            self._model_info = {
                '1P': mp.Value(ModelInfo, 'default', self._first_ckpt),
            }
        elif self._num_players == 2:
            self._model_info = {
                '1P': mp.Value(ModelInfo, 'default', self._first_ckpt),
                '2P': mp.Value(ModelInfo, 'default', self._first_ckpt),
            }
        # tensorboard log
        storage_path = Path(args.storage_dir)
        log_path = storage_path / 'log' / 'eval'
        if not log_path.exists():
            log_path.mkdir(exist_ok=True)
        self._summary_writer = SummaryWriter(log_dir=log_path, purge_step=self._first_ckpt)

        # start EvalEnvManager
        self._num_env_per_proc = args.num_env * self._num_players
        self._total_env = self._num_env_per_proc * args.num_proc
        self._job_queue = Queue()
        self._result_queue = Queue()
        self._pipes = [mp.Pipe() for i in range(args.num_proc)]
        self._managers = [
            mp.Process(
                target=lambda *args: asyncio.run(run_eval_env_manager(*args)),
                args=(
                    args,
                    config,
                    callbacks,
                    index,
                    pipe,
                    self._job_queue,
                    self._result_queue,
                    self._operation,
                    self._model_info,
                ),
            ) for index, (_, pipe) in enumerate(self._pipes)
        ]
        for manager in self._managers:
            manager.start()

        # connect to model info upstream
        print('connect to learner/model_provider @', args.upstream)
        self._upstream = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.upstream,
        )
        # subscribe for the latest model info
        asyncio.create_task(self._send_model_subscribe())
        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )

    def terminate(self):
        '''terminate all EvalEnvManager'''
        for manager in self._managers:
            manager.join()

    async def loop(self):
        '''main loop'''
        await asyncio.gather(
            self._recv_job_loop(),
            self._recv_model_info_loop(),
            self._send_job_loop(),
            self._eval_loop(),
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
                    for pipe in self._pipes:
                        pipe[0].send(job.SerializeToString())
                    continue
                index = job.payload.env_index // self._num_env_per_proc
                self._pipes[index][0].send(job.SerializeToString())
            elif packet_type == 'job_batch':
                jobs = packet.job_batch.jobs
                for job in jobs:
                    if not job.HasField('payload'):
                        # if flush model has done, start jobs for each env
                        for pipe in self._pipes:
                            pipe[0].send(job.SerializeToString())
                        continue
                    index = job.payload.env_index // self._num_env_per_proc
                    self._pipes[index][0].send(job.SerializeToString())

    async def _recv_model_info_loop(self):
        '''a loop to receive `ModelInfo`'''
        while True:
            raw = await self._upstream.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_info':
                new_version = packet.model_info.version
                if self._latest and new_version > self._last_ckpt:
                    for version in range(self._last_ckpt + 1, new_version + 1):
                        if (version - self._first_ckpt) % self._freq == 0:
                            self._model_version_queue.put(version)
                    self._last_ckpt = new_version

    async def _send_job_loop(self):
        '''a loop to send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._job_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            await self._broker.send(raw)

    async def _eval_loop(self):
        executor = ThreadPoolExecutor(max_workers=1)
        result_executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._model_version_queue
        result_queue = self._result_queue
        while True:
            # get model version and send flush job
            version = await loop.run_in_executor(executor, queue.get)
            if self._num_players == 2 and version == self._first_ckpt:
                continue
            self._model_info['1P'].version = version
            await self._send_load_model()
            # wait for the results of all envs
            eval_results = []
            while len(eval_results) < self._total_env:
                result = await loop.run_in_executor(result_executor, result_queue.get)
                eval_results.append(result)
            asyncio.create_task(self._write_result(eval_results))

    async def _write_result(self, eval_results: list):
        '''helper to process evaluation result'''
        total_env = len(eval_results)
        assert total_env == self._total_env
        game_steps, total_rewards = [], []
        for result in eval_results:
            game_steps.append(result.game_steps)
            total_rewards.append(result.total_rewards)

        step = self._model_info['1P'].version
        writer = self._summary_writer
        writer.add_scalars(
            'eval/steps', {
                'mean': np.mean(game_steps),
                'min': np.min(game_steps),
                'max': np.max(game_steps),
                'std': np.std(game_steps),
            }, step)

        if self._num_players == 1:
            print('{} Reward Mean: {:.2f}, Max: {:.2f}, Min: {:.2f}'.format(
                self._model_info['1P'].version,
                np.mean(total_rewards),
                np.max(total_rewards),
                np.min(total_rewards),
            ))
            writer.add_scalars(
                'eval/score', {
                    'mean': np.mean(total_rewards),
                    'min': np.min(total_rewards),
                    'max': np.max(total_rewards),
                    'std': np.std(total_rewards),
                }, step)
            writer.flush()
        elif self._num_players == 2:
            win_rate = total_rewards.count(1) / total_env
            draw_rate = total_rewards.count(0) / total_env
            lose_rate = total_rewards.count(-1) / total_env
            # score for win: 2, draw: 1, lose: 0
            score = np.sum(np.array(total_rewards) + 1) / (2 * total_env)
            elo_1p_diff = 400 * np.log10(score / (1 - score))
            elo_1p = elo_1p_diff + self._best_elo
            print(
                '{}-{} Win: {:.2%} Draw: {:.2%} Lose: {:.2%} Current Elo: {:.1f} ({:+.1f})'.format(
                    self._model_info['1P'].version,
                    self._model_info['2P'].version,
                    win_rate,
                    draw_rate,
                    lose_rate,
                    elo_1p,
                    elo_1p_diff,
                ))
            # write the evaluation result to tensorboard
            writer.add_scalar('eval/elo', elo_1p, step)
            writer.add_scalar('eval/current_version', self._model_info['1P'].version, step)
            writer.add_scalar('eval/best_version', self._model_info['2P'].version, step)
            writer.add_scalars('eval/result', {
                'win': win_rate,
                'draw': draw_rate,
                'lose': lose_rate,
            }, step)
            writer.flush()
            # update model according to the score
            if score > self._replace_rate:  # current model (1p) is the best
                # self._model_info['2P'].name = self._model_info['1P'].name
                self._model_info['2P'].version = self._model_info['1P'].version
                self._best_elo = elo_1p

    async def _send_load_model(self):
        '''helper to wait for new model and send jobs to flush'''
        # send jobs to flush model
        for (_, op), (_, model_info) in zip(self._operation.items(), self._model_info.items()):
            job = czf_pb2.Job(
                model=czf_pb2.ModelInfo(
                    name=model_info.name,
                    version=model_info.version,
                ),
                procedure=[op],
                step=0,
            )
            job.initiator.CopyFrom(self._node)
            packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job]))
            # print(packet)
            await self._broker.send(packet.SerializeToString())

    async def _send_model_subscribe(self):
        '''helper to send a `model_subscribe` to optimizer'''
        packet = czf_pb2.Packet(model_subscribe=czf_pb2.Heartbeat())
        await self._upstream.send(packet.SerializeToString())
