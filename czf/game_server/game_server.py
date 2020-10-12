'''CZF Game Server'''
import asyncio
import platform
import random
import zmq
import zmq.asyncio
import czf_env

from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer


class EnvManager:
    '''Game Environment Manager'''
    def __init__(self, server):
        self.server = server
        self.reset()
        asyncio.create_task(self.send_search_job_request())

    def reset(self):
        '''reset envs'''
        self.state = self.server.game.new_initial_state()
        self.trajectory = czf_pb2.Trajectory()
        self.trajectory.returns[:] = [0.0] * self.server.game.num_players

        self.workers = [czf_pb2.Node()] * 1
        self.model = czf_pb2.Model(
            name=self.server.model_name,
            version=-1,
        )

    async def send_search_job_request(
        self,
        actor=czf_pb2.Job.Operation.MUZERO_SEARCH,
    ):
        '''helper to send a `Job` to actor'''
        job = czf_pb2.Job(
            model=self.model,
            procedure=[actor],
            step=0,
            workers=self.workers,
            payload=czf_pb2.Job.Payload(
                state=czf_pb2.State(
                    serialize=self.state.serialize(),
                    legal_actions=self.state.legal_actions,
                    observation_tensor=self.state.observation_tensor,
                ),
                env_index=self.server.envs.index(self),
            ))
        await self.server.send_job(job)

    def on_job_completed(self, job: czf_pb2.Job):
        '''callback on job completion'''
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
    '''Game Server'''
    def __init__(self, args):
        self.node = czf_pb2.Node(
            hostname=platform.node(),
            identity=f'game-server-{args.suffix}',
        )
        self.model_name = 'default'
        # game envs
        self.game = czf_env.load_game(args.game)
        self.envs = [EnvManager(self) for _ in range(args.num_env)]
        # connect to broker
        self.socket = get_zmq_dealer(
            identity=self.node.identity,
            remote_address=args.broker,
        )

    async def loop(self):
        '''main loop'''
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            print(packet)
            if packet_type == 'job':
                job = packet.job
                env_index = job.payload.env_index
                self.envs[env_index].on_job_completed(job)

    async def send_job(self, job: czf_pb2.Job):
        '''helper to send a `Job`'''
        job.initiator.CopyFrom(self.node)
        packet = czf_pb2.Packet(job=job)
        await self.__send_packet(packet)

    async def send_optimize_job(
            self,
            trajectory: czf_pb2.Trajectory,
            optimizor=czf_pb2.Job.Operation.MUZERO_OPTIMIZE):
        '''helper to send a `Job` to optimizor'''
        job = czf_pb2.Job(
            procedure=[optimizor],
            step=0,
            workers=[czf_pb2.Node()],
            payload=czf_pb2.Job.Payload(trajectory=trajectory),
        )
        await self.send_job(job)

    async def __send_packet(self, packet: czf_pb2.Packet):
        '''helper to send a `Packet`'''
        await self.socket.send(packet.SerializeToString())
