'''CZF Game Server'''
import asyncio
import platform
import random
import czf_env
import numpy as np

from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer


class EnvManager:
    '''Game Environment Manager'''
    def __init__(self, server):
        self._server = server
        self._zero_policy = np.zeros(self._server.game.num_distinct_actions)
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
                state=czf_pb2.State(
                    legal_actions=self._state.legal_actions,
                    observation_tensor=self._state.observation_tensor,
                    serialize=self._state.serialize(),
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
        chosen_action = -1
        if num_moves < 2:
            # softmax move
            chosen_action = random.choices(legal_actions,
                                           legal_actions_policy)[0]
        else:
            # argmax move
            chosen_action = legal_actions[np.argmax(legal_actions_policy)]
        # add to trajectory
        state = self._trajectory.states.add()
        state.observation_tensor[:] = evaluated_state.observation_tensor
        state.evaluation.policy[:] = policy
        state.transition.current_player = self._state.current_player
        state.transition.action = chosen_action
        # apply action
        self._state.apply_action(chosen_action)
        # game transition
        state.transition.rewards[:] = self._state.rewards

        if self._state.is_terminal:
            asyncio.create_task(
                self._server.send_optimize_job(self._trajectory))
            self.reset()

        if len(self._state.legal_actions) == 1:
            # choose the only action
            policy = self._zero_policy[:]
            policy[self._state.legal_actions[0]] = 1.
            job = czf_pb2.Job(payload=czf_pb2.Job.Payload(
                state=czf_pb2.State(
                    observation_tensor=self._state.observation_tensor,
                    evaluation=czf_pb2.State.Evaluation(policy=list(policy)),
                ),
                env_index=self._server.envs.index(self),
            ))
            asyncio.create_task(self.on_job_completed(job))
        else:
            # send a search job
            asyncio.create_task(self.send_search_job_request())


class GameServer:
    '''Game Server'''
    def __init__(self, args):
        self._node = czf_pb2.Node(
            identity=f'game-server-{args.suffix}',
            hostname=platform.node(),
        )
        self.model_name = 'default'
        # game envs
        self.game = czf_env.load_game(args.game)
        self.envs = [EnvManager(self) for _ in range(args.num_env)]
        # connect to broker
        self._socket = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )

    async def loop(self):
        '''main loop'''
        while True:
            raw = await self._socket.recv()
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
        await self._socket.send(packet.SerializeToString())
