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

from czf.env import czf_env, atari_env
from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer, Queue


async def run_env_manager(*args):
    '''run :class:`EnvManager`'''
    manager = EnvManager(*args)
    manager.start_search()
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


class EnvManager:
    '''Game Environment Manager'''
    def __init__(self, args, config, callbacks, proc_index, model_info, pipe, job_queue,
                 trajectory_queue):
        self._algorithm = config['algorithm']
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}', hostname=platform.node())
        # multiprocess
        self._proc_index = proc_index
        self._model_info = model_info
        self._pipe = pipe
        self._job_queue = job_queue
        self._trajectory_queue = trajectory_queue
        # callbacks
        self._action_policy_fn = callbacks['action_policy']
        metric_callbacks = callbacks.get('metric', {})
        self._after_apply_callback = metric_callbacks.get('after_apply', None)
        # tree config
        mcts_config = config['mcts']
        if self._algorithm == 'AlphaZero':
            self._operation = czf_pb2.Job.Operation.ALPHAZERO_SEARCH
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=mcts_config['dirichlet']['epsilon'],
            )
        elif self._algorithm == 'MuZero':
            self._operation = czf_pb2.Job.Operation.MUZERO_SEARCH
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                tree_min_value=mcts_config.get('tree_min_value', float('inf')),
                tree_max_value=mcts_config.get('tree_max_value', float('-inf')),
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=mcts_config['dirichlet']['epsilon'],
                discount=mcts_config.get('discount', 1.),
            )
            kstep = config['learner']['rollout_steps']
            nstep = mcts_config['nstep']
            self._mstep = kstep + nstep
        # game_server config
        self._sequence = config['game_server']['sequence']
        # game env
        self._num_env = args.num_env
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
        self._envs = [None] * args.num_env
        self._total_rewards = [None] * self._num_env
        self._num_steps = [None] * self._num_env
        for index in range(args.num_env):
            self.__reset(index, new=True)

    def start_search(self):
        '''all envs start to send search'''
        for index in range(len(self._envs)):
            self.__send_search_job(index)

    async def loop(self):
        '''main loop'''
        while True:
            job = self._pipe.recv()
            self.__on_job_completed(czf_pb2.Job.FromString(job))

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
        if self._frame_stack > 1:
            env = self._envs[env_index]
            for _ in range(self._frame_stack - 1):
                state = env.trajectory.states.add()
                state.observation_tensor[:] = env.state.feature_tensor
        self._total_rewards[env_index] = [0.] * self._game.num_players
        self._num_steps[env_index] = 0

    def __send_search_job(self, env_index):
        '''helper to send a `Job` to actor'''
        env = self._envs[env_index]
        # workers = [czf_pb2.Node(identity='g', hostname=str(time.time()))] * 2
        if self._algorithm == 'AlphaZero':
            state = czf_pb2.WorkerState(serialized_state=env.state.serialize(), )
        elif self._algorithm == 'MuZero':
            state = czf_pb2.WorkerState(
                legal_actions=env.state.legal_actions,
                observation_tensor=env.state.observation_tensor,
            )
        job = czf_pb2.Job(model=czf_pb2.ModelInfo(name=self._model_info.name,
                                                  version=self._model_info.version),
                          procedure=[self._operation],
                          step=0,
                          payload=czf_pb2.Job.Payload(env_index=self._proc_index * self._num_env +
                                                      env_index, ))
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
        chosen_action = self._action_policy_fn(num_moves, legal_actions, legal_actions_policy)
        # add to trajectory
        state = env.trajectory.states.add()
        state.observation_tensor[:] = env.state.feature_tensor
        state.evaluation.value = evaluated_state.evaluation.value
        state.evaluation.policy[:] = policy
        state.transition.current_player = env.state.current_player
        state.transition.action = chosen_action
        # apply action
        # print(job.workers, time.time())
        # print('apply', self._proc_index, env_index,
        #       time.time() - self.start_time[env_index])
        env.state.apply_action(chosen_action)
        self._num_steps[env_index] += 1
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state, env.state)
        # game transition
        state.transition.rewards[:] = env.state.rewards
        for player, reward in enumerate(env.state.rewards):
            self._total_rewards[env_index][player] += reward

        if env.state.is_terminal:
            # (MuZero) add the terminal state to the trajectory
            if self._algorithm == 'MuZero':
                state = env.trajectory.states.add()
                state.observation_tensor[:] = env.state.feature_tensor
                state.evaluation.value = evaluated_state.evaluation.value
                state.transition.current_player = env.state.current_player
                state.transition.rewards[:] = env.state.rewards
            # add game statistics
            env.trajectory.statistics.rewards[:] = self._total_rewards[env_index]
            env.trajectory.statistics.game_steps = self._num_steps[env_index]
            # send optimize job
            print('send 1 traj with len', len(env.trajectory.states), 'of score',
                  int(self._total_rewards[env_index][0]))
            packet = czf_pb2.Packet(trajectory_batch=czf_pb2.TrajectoryBatch(
                trajectories=[env.trajectory]))
            self._trajectory_queue.put(packet.SerializeToString())
            self.__reset(env_index)
        elif self._sequence > 0 and (len(env.trajectory.states) %
                                     (self._sequence + self._mstep) == 0):
            # send optimize job of length (sequence + mstep)
            print('send 1 traj of len', len(env.trajectory.states))
            packet = czf_pb2.Packet(trajectory_batch=czf_pb2.TrajectoryBatch(
                trajectories=[env.trajectory]))
            self._trajectory_queue.put(packet.SerializeToString())
            # keep the last (frame_stack - 1 + mstep) states
            trajectory = czf_pb2.Trajectory()
            if self._frame_stack > 1:
                for i in reversed(range(self._frame_stack - 1)):
                    state = trajectory.states.add()
                    state.observation_tensor[:] = env.trajectory.states[
                        -i - 1 - self._mstep].observation_tensor
            trajectory.states.extend(env.trajectory.states[-self._mstep:])
            env.trajectory = trajectory

        # send a search job
        self.__send_search_job(env_index)


class GameServer:
    '''Game Server'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}', hostname=platform.node())
        # model
        self._model_info = mp.Value(ModelInfo, 'default', -1)
        self._has_new_model = asyncio.Event()
        # server mode
        algorithm = config['algorithm']
        assert algorithm in ('AlphaZero', 'MuZero')
        print(f'[{algorithm} Training Mode]', self._node.identity)

        # start EnvManager
        self._num_env = args.num_env
        self._job_queue = Queue()
        self._trajectory_queue = Queue()
        self._pipe = [mp.Pipe() for i in range(args.num_proc)]
        self._manager = [
            mp.Process(target=lambda *args: asyncio.run(run_env_manager(*args)),
                       args=(
                           args,
                           config,
                           callbacks,
                           index,
                           self._model_info,
                           pipe,
                           self._job_queue,
                           self._trajectory_queue,
                       )) for index, (_, pipe) in enumerate(self._pipe)
        ]
        for manager in self._manager:
            manager.start()

        # trajectory upstream
        print('connect to learner @', args.upstream)
        self._upstream = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.upstream,
        )
        asyncio.create_task(self.__send_model_subscribe())
        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )

    def terminate(self):
        '''terminate all EnvManager'''
        for manager in self._manager:
            manager.join()

    async def loop(self):
        '''main loop'''
        await asyncio.gather(
            self._recv_job_loop(),
            self._recv_model_info_loop(),
            self._send_job_loop(),
            self._send_trajectory_loop(),
        )

    async def _recv_job_loop(self):
        '''a loop to receive `Job`'''
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            #print(packet)
            if packet_type == 'job':
                job = packet.job
                index = job.payload.env_index // self._num_env
                self._pipe[index][0].send(job.SerializeToString())
            elif packet_type == 'job_batch':
                jobs = packet.job_batch.jobs
                for job in jobs:
                    index = job.payload.env_index // self._num_env
                    self._pipe[index][0].send(job.SerializeToString())

    async def _recv_model_info_loop(self):
        '''a loop to receive `ModelInfo`'''
        while True:
            raw = await self._upstream.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_info':
                version = packet.model_info.version
                # assert packet._model_info.name == self._model_info.name
                if version > self._model_info.version:
                    self._model_info.version = version
                    self._has_new_model.set()

    async def _send_job_loop(self):
        '''a loop to send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._job_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            await self._broker.send(raw)

    async def _send_trajectory_loop(self):
        '''a loop to send `Trajectory`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._trajectory_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            await self._upstream.send(raw)

    async def __send_model_subscribe(self):
        '''helper to send a `model_subscribe` to optimizer'''
        packet = czf_pb2.Packet(model_subscribe=czf_pb2.Heartbeat())
        await self._upstream.send(packet.SerializeToString())
