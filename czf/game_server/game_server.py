'''CZF Game Server'''
import asyncio
import platform
import random
import zmq
import zmq.asyncio
import czf_env

from czf.pb import czf_pb2


class EnvManager:
    def __init__(self, server):
        self.server = server
        self.reset()
        asyncio.create_task(self.send_search_job_request())

    def reset(self):
        self.state = self.server.game.new_initial_state()
        self.trajectory = czf_pb2.Trajectory()
        self.trajectory.returns[:] = [0.0] * self.server.game.num_players

        self.workers = [czf_pb2.Node()] * 1
        self.model = czf_pb2.Model(name='resnet', version=-1)

    async def send_search_job_request(self):
        job = czf_pb2.Job(
            model=self.model,
            procedure=[czf_pb2.Job.Operation.ALPHAZERO_SEARCH],
            step=0,
            workers=self.workers,
            payload=czf_pb2.Job.Payload(
                state=czf_pb2.State(
                    serialize=self.state.serialize(),
                    legal_actions=self.state.legal_actions,
                    observation_tensor=self.state.observation_tensor,
                    observation_tensor_shape=self.server.game.
                    observation_tensor_shape),
                env_index=self.server.envs.index(self)))
        await self.server.send_job(job)

    def on_job_completed(self, job: czf_pb2.Job):
        self.workers[:] = job.workers
        self.model.CopyFrom(job.model)

        evaluated_state = job.payload.state
        policy = evaluated_state.evaluation.policy
        legal_actions_policy = [
            policy[action] for action in self.state.legal_actions
        ]
        chosen_action = random.choices(self.state.legal_actions,
                                       legal_actions_policy)[0]

        state = self.trajectory.states.add()
        state.current_player = self.state.current_player
        state.observation_tensor[:] = self.state.observation_tensor
        state.evaluation.policy[:] = policy

        self.state.apply_action(chosen_action)
        state.transition.action = chosen_action
        state.transition.rewards[:] = self.state.rewards

        for i in range(self.server.game.num_players):
            self.trajectory.returns[i] += state.transition.rewards[i]

        if self.state.is_terminal:
            asyncio.create_task(self.server.send_optimize_job(self.trajectory))
            self.reset()

        asyncio.create_task(self.send_search_job_request())


class GameServer:
    def __init__(self, args):
        self.node = czf_pb2.Node(hostname=platform.node(),
                                 identity=f'game-server-{args.suffix}')
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.IDENTITY, self.node.identity)
        socket.connect(f'tcp://{args.broker}')
        self.socket = socket

        self.game = czf_env.load_game(args.game)
        self.envs = [EnvManager(self) for _ in range(args.num_env)]

    async def loop(self):
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            print(packet)
            if packet_type == 'job':
                job = packet.job
                env_index = job.payload.env_index
                self.envs[env_index].on_job_completed(job)

    async def send_packet(self, packet: czf_pb2.Packet):
        raw = packet.SerializeToString()
        await self.socket.send(raw)

    async def send_job(self, job: czf_pb2.Job):
        job.initiator.CopyFrom(self.node)
        packet = czf_pb2.Packet(job=job)
        await self.send_packet(packet)

    async def send_optimize_job(self, trajectory: czf_pb2.Trajectory):
        job = czf_pb2.Job(procedure=[czf_pb2.Job.Operation.ALPHAZERO_OPTIMIZE],
                          step=0,
                          workers=[czf_pb2.Node()],
                          payload=czf_pb2.Job.Payload(trajectory=trajectory))
        await self.send_job(job)
