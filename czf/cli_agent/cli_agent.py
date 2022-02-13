'''CZF CLI Agent'''
import asyncio
import platform
import sys
import time

from czf.env import czf_env
from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer


class CliAgent:
    '''CLI Agent'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(
            identity=f'cli-agent-{args.suffix}',
            hostname=platform.node(),
        )
        # callbacks
        self._action_policy_fn = callbacks['action_policy']
        metric_callbacks = callbacks.get('metric', {})
        self._after_apply_callback = metric_callbacks.get('after_apply', None)
        # model
        self._model_info = czf_pb2.ModelInfo(
            name='default',
            version=args.model_version,
        )
        # tree config
        mcts_config = config['mcts']
        self._algorithm = config['algorithm']
        if self._algorithm == 'AlphaZero':
            self._operation = czf_pb2.Job.Operation.ALPHAZERO_SEARCH
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=0.,
            )
        elif self._algorithm == 'MuZero':
            self._operation = czf_pb2.Job.Operation.MUZERO_SEARCH
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                tree_min_value=mcts_config.get('tree_min_value', float('inf')),
                tree_max_value=mcts_config.get('tree_max_value', float('-inf')),
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=0.,
                discount=mcts_config.get('discount', 1.),
            )
        elif self._algorithm == 'MuZero_Gumbel':
            self._operation = czf_pb2.Job.Operation.MUZERO_SEARCH
            self._tree_option = czf_pb2.WorkerState.TreeOption(
                simulation_count=mcts_config['simulation_count'],
                tree_min_value=mcts_config.get('tree_min_value', float('inf')),
                tree_max_value=mcts_config.get('tree_max_value', float('-inf')),
                c_puct=mcts_config['c_puct'],
                dirichlet_alpha=mcts_config['dirichlet']['alpha'],
                dirichlet_epsilon=0.,
                discount=mcts_config.get('discount', 1.),
                gumbel_sampled_actions=mcts_config['gumbel']['sampled_actions'],
                gumbel_c_visit=mcts_config['gumbel']['c_visit'],
                gumbel_c_scale=mcts_config['gumbel']['c_scale'],
                gumbel_use_noise=False,
            )
        # game env
        game_config = config['game']
        obs_config = game_config['observation']
        env_name = game_config['name']
        if env_name in czf_env.available_games():
            self._game = czf_env.load_game(env_name)
            # check game config in lightweight game
            assert self._game.num_players == game_config['num_player']
            assert self._game.num_distinct_actions == game_config['actions']
            assert self._game.observation_tensor_shape == [
                obs_config['channel'], *obs_config['spatial_shape']
            ]

        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )
        self._flush = False

    async def send_load_model(self):
        '''helper to wait for new model and send jobs to flush'''
        # send job to flush model
        job = czf_pb2.Job(procedure=[self._operation], step=0)
        job.model.CopyFrom(self._model_info)
        job.initiator.CopyFrom(self._node)
        await self._send_job(job)

        # receive job to ensure flushing is finished
        job = await self._recv_job()
        if not job.HasField('payload'):
            self._flush = True

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._cmd_loop())

    async def _cmd_loop(self):
        '''command loop'''
        self._reset()
        self._showboard()
        while not self._state.is_terminal:
            print('$ ', end='')
            command = input()
            if 'showboard' in command or 'sb' in command:
                self._showboard()
            elif 'clear' in command or 'reset' in command:
                self._reset()
            elif 'genmove' in command or 'gen' in command:
                if not self._flush:
                    print('Model flushing is not finished yet')
                    continue
                if len(command.split()) > 1:
                    self._tree_option.simulation_count = int(command.split()[-1])
                actions = await self._genmove()
                self._play(actions)
                actions_string = [self._game.action_to_string(action) for action in actions]
                print('action: ' + ', '.join(actions_string))
                self._history.append(actions_string)
            elif 'play' in command:
                substr = command[command.find('play') + len('play'):]
                actions = self._game.string_to_action(substr)
                self._play(actions)
                actions_string = [self._game.action_to_string(action) for action in actions]
                print('action: ' + ', '.join(actions_string))
                self._history.append(actions_string)
            elif 'exit' in command or 'quit' in command:
                break
            self._showboard()

        print(f'Final rewards: {self._state.rewards}')
        print(f'Total steps: {self._num_steps}')

    def _showboard(self, file=sys.stdout):
        print(self._state, file=file)
        print(file=file)

    def _reset(self):
        self._state = self._game.new_initial_state()
        self._history = []
        self._num_steps = 0

    async def _genmove(self):
        # send job
        # print('serialize:', self._state.serialize())
        if self._algorithm == 'AlphaZero':
            state = czf_pb2.WorkerState(serialized_state=self._state.serialize())
        elif self._algorithm == 'MuZero' or self._algorithm == 'MuZero_Gumbel':
            state = czf_pb2.WorkerState(
                legal_actions=self._state.legal_actions,
                observation_tensor=self._state.observation_tensor,
            )
        job = czf_pb2.Job(
            model=czf_pb2.ModelInfo(
                name=self._model_info.name,
                version=self._model_info.version,
            ),
            procedure=[self._operation],
            step=0,
            payload=czf_pb2.Job.Payload(),
        )
        job.initiator.CopyFrom(self._node)
        # job.payload.state.workers.CopyFrom(env.workers)
        job.payload.state.CopyFrom(state)
        job.payload.state.tree_option.CopyFrom(self._tree_option)
        await self._send_job(job)

        # receive job
        job = await self._recv_job()
        evaluated_state = job.payload.state
        policy = evaluated_state.evaluation.policy
        legal_actions = self._state.legal_actions
        legal_actions_policy = [policy[action] for action in legal_actions]
        num_moves = self._num_steps
        chosen_action = self._action_policy_fn(
            num_moves,
            legal_actions,
            legal_actions_policy,
        )
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state)
        return [chosen_action]

    def _play(self, actions):
        for action in actions:
            if action not in self._state.legal_actions:
                print('Illegal Action')
            else:
                self._num_steps += 1
                self._state.apply_action(action)

    async def _send_job(self, job):
        packet = czf_pb2.Packet(job=job)
        await self._broker.send(packet.SerializeToString())

    async def _recv_job(self):
        raw = await self._broker.recv()
        packet = czf_pb2.Packet.FromString(raw)
        packet_type = packet.WhichOneof('payload')
        if packet_type == 'job':
            job = packet.job
        elif packet_type == 'job_batch':
            job = packet.job_batch.jobs[0]
        return job
