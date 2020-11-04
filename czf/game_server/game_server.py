'''CZF Game Server'''
import asyncio
import platform
import czf_env

from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer


class EnvManager:
    '''Game Environment Manager'''
    def __init__(self, server):
        self._server = server
        self._action_policy_fn = self._server.action_policy_fn
        self._after_apply_callback = self._server.metric_callbacks.get(
            'after_apply', None)
        self.reset()
        asyncio.create_task(self.send_search_job_request())

    def reset(self):
        '''reset envs'''
        self._state = self._server.game.new_initial_state()
        self._trajectory = czf_pb2.Trajectory()
        self._workers = [czf_pb2.Node()] * 1
        self._model = czf_pb2.ModelInfo(
            name=self._server.model_name,
            version=-1,
        )

    async def send_search_job_request(
        self,
        actor=czf_pb2.Job.Operation.MUZERO_SEARCH,
    ):
        '''helper to send a `Job` to actor'''
        job = czf_pb2.Job(
            model=self._model,
            procedure=[actor],
            step=0,
            workers=self._workers,
            payload=czf_pb2.Job.Payload(
                state=czf_pb2.WorkerState(
                    legal_actions=self._state.legal_actions,
                    observation_tensor=self._state.observation_tensor,
                    tree_option=self._server.tree_option,
                ),
                env_index=self._server.envs.index(self),
            ))
        await self._server.send_job(job)

    async def on_job_completed(self, job: czf_pb2.Job):
        '''callback on job completion'''
        self._workers[:] = job.workers
        self._model.CopyFrom(job.model)

        evaluated_state = job.payload.state
        # choose action according to the policy
        policy = evaluated_state.evaluation.policy
        legal_actions = self._state.legal_actions
        legal_actions_policy = [policy[action] for action in legal_actions]
        num_moves = len(self._trajectory.states)
        chosen_action = self._action_policy_fn(num_moves, legal_actions,
                                               legal_actions_policy)
        # add to trajectory
        state = self._trajectory.states.add()
        state.observation_tensor[:] = evaluated_state.observation_tensor
        state.tree_option.CopyFrom(self._server.tree_option)
        state.evaluation.policy[:] = policy
        state.transition.current_player = self._state.current_player
        state.transition.action = chosen_action
        # apply action
        self._state.apply_action(chosen_action)
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state, self._state)
        # game transition
        state.transition.rewards[:] = self._state.rewards

        if self._state.is_terminal:
            # add the terminal state to the trajectory
            state = self._trajectory.states.add()
            state.observation_tensor[:] = self._state.observation_tensor
            state.evaluation.value = 0
            state.transition.current_player = self._state.current_player
            state.transition.rewards[:] = self._state.rewards
            # send optimize job
            asyncio.create_task(
                self._server.send_optimize_job(self._trajectory))
            self.reset()

        # send a search job
        asyncio.create_task(self.send_search_job_request())


class GameServer:
    '''Game Server'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(
            identity=f'game-server-{args.suffix}',
            hostname=platform.node(),
        )
        self.model_name = 'default'
        # server mode
        self._is_evaluation = args.eval
        if args.eval:
            print('[Evaluation Mode]', self._node.identity)
        else:
            print('[Training Mode]', self._node.identity)
        # callbacks
        self.action_policy_fn = callbacks['action_policy']
        self.metric_callbacks = callbacks.get('metric', {})
        if self.metric_callbacks:
            print('register callbacks:', list(self.metric_callbacks.keys()))
        # game envs
        game_config = config['game']
        self.game = czf_env.load_game(game_config['name'])
        self.envs = [EnvManager(self) for _ in range(args.num_env)]
        # check game config in lightweight game
        assert self.game.num_players == game_config['num_player']
        assert self.game.num_distinct_actions == game_config['actions']
        assert self.game.observation_tensor_shape == game_config[
            'observation_shape']
        # tree config
        mcts_config = config['mcts']
        self.tree_option = czf_pb2.WorkerState.TreeOption(
            simulation_count=mcts_config['simulation_count'],
            tree_min_value=mcts_config.get('tree_min_value', float('inf')),
            tree_max_value=mcts_config.get('tree_max_value', float('-inf')),
            c_puct=mcts_config['c_puct'],
            dirichlet_alpha=mcts_config['dirichlet']['alpha'],
            dirichlet_epsilon=mcts_config['dirichlet']['epsilon'],
            discount=mcts_config.get('discount', 1.),
        )
        if args.eval:
            self.tree_option.dirichlet_epsilon = 0.
        # trajectory upstream
        print('connect to learner @', args.upstream)
        self._upstream = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.upstream,
        )
        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )

    async def loop(self):
        '''main loop'''
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            #print(packet)
            if packet_type == 'job':
                job = packet.job
                env_index = job.payload.env_index
                asyncio.create_task(self.envs[env_index].on_job_completed(job))

    async def send_job(self, job: czf_pb2.Job):
        '''helper to send a `Job`'''
        job.initiator.CopyFrom(self._node)
        packet = czf_pb2.Packet(job=job)
        await self.__send_packet(packet)

    async def send_optimize_job(self, trajectory: czf_pb2.Trajectory):
        '''helper to send a `Trajectory` to optimizer'''
        if not self._is_evaluation:
            packet = czf_pb2.Packet(trajectory_batch=czf_pb2.TrajectoryBatch(
                trajectories=[trajectory]))
            await self._upstream.send(packet.SerializeToString())

    async def __send_packet(self, packet: czf_pb2.Packet):
        '''helper to send a `Packet`'''
        await self._broker.send(packet.SerializeToString())
