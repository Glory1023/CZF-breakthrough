'''CZF Evaluator'''
import asyncio
import numpy as np

from czf.game_server.game_server import GameServer, EnvManager
from czf.pb import czf_pb2


class EvalEnvManager(EnvManager):
    '''Game Evaluation Environment Manager'''
    async def send_search_job(self):
        '''helper to send a `Job` to actor'''
        player = self._state.current_player
        models = [self._server.model_1p, self._server.model_2p]
        job = czf_pb2.Job(
            model=models[player],
            procedure=[self.operation[player]],
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
        # choose action according to the policy
        evaluated_state = job.payload.state
        policy = evaluated_state.evaluation.policy
        legal_actions = self._state.legal_actions
        legal_actions_policy = [policy[action] for action in legal_actions]
        num_moves = len(self._trajectory.states)
        chosen_action = self._action_policy_fn(num_moves, legal_actions,
                                               legal_actions_policy)
        # add to trajectory
        self._trajectory.states.add()
        # apply action
        self._state.apply_action(chosen_action)
        if self._after_apply_callback:
            self._after_apply_callback(evaluated_state, self._state)

        if self._state.is_terminal:
            self.reset()
            first_player = self.operation[0]
            if first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_1P:
                reward = self._state.rewards[0]
            else:  # first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_2P:
                reward = self._state.rewards[1]
            return True, reward
        return False, None


class EvalGameServer(GameServer):
    '''Evalutation Game Server'''
    def __init__(self, args, config, callbacks):
        super().__init__(args, config, callbacks)
        eval_config = config['evaluator']
        assert eval_config['mode'] == 'best'
        self._freq = eval_config['frequency']
        self._total = args.num_env
        self._workers = [czf_pb2.Node()]
        # 2 * total envs (1P/2P exchange)
        self.envs = [
            EvalEnvManager(self, start=False) for _ in range(2 * self._total)
        ]
        for i in range(self._total):
            self.envs[i].operation = [
                czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
            ]
            self.envs[self._total + i].operation = [
                czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
                czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
            ]
        # in best mode, 1p is the latest, 2p is the best
        self.model_1p = czf_pb2.ModelInfo(name='default', version=0)
        self.model_2p = czf_pb2.ModelInfo(name='default', version=0)
        asyncio.create_task(self.__send_load_model())

    async def _job_loop(self):
        '''a loop to receive `Job`'''
        eval_result = []
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            #print(packet)
            if packet_type == 'job':
                job = packet.job
                if not job.HasField('payload'):
                    # if flush model has done, start jobs for each env
                    operation = job.procedure[0]
                    for env in self.envs:
                        if env.operation[0] == operation:
                            asyncio.create_task(env.send_search_job())
                    continue
                env = self.envs[job.payload.env_index]
                done, result = await env.on_job_completed(job)
                if done:
                    eval_result.append(result)
                    if len(eval_result) == self._total:
                        asyncio.create_task(self.__write_result(eval_result))
                        eval_result = []
                    else:
                        asyncio.create_task(env.send_search_job())
                else:
                    asyncio.create_task(env.send_search_job())

    async def __write_result(self, eval_result):
        '''helper to process evaluation result'''
        # TODO: write to tensorboard
        total = len(eval_result)
        win_rate = eval_result.count(1) / total
        draw_rate = eval_result.count(0) / total
        lose_rate = eval_result.count(-1) / total
        print('Win: {:.2%} Draw: {:.2%} Lose: {:.2%}'.format(
            win_rate, draw_rate, lose_rate))
        # score for win: 2, draw: 1, lose: 0
        score = np.sum(np.array(eval_result) + 1) / (2 * total)
        if score > 0.55:  # current model (1p) is the best
            self.model_2p.CopyFrom(self.model_1p)
        await self.__send_load_model()

    async def __send_load_model(self):
        '''helper to wait for new model and send jobs to flush'''
        # wait for next model version
        next_version = self.model_1p.version + self._freq
        if self.model.version < next_version:
            while await self.has_new_model.wait():
                self.has_new_model.clear()
                if self.model.version >= next_version:
                    self.model_1p.version = next_version
                    break
        else:
            self.model_1p.version = next_version
        print('send 1p', self.model_1p.version, '2p', self.model_2p.version)
        # send jobs to flush model
        job1 = czf_pb2.Job(
            model=self.model_1p,
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_1P],
            step=0,
            workers=self._workers,
        )
        await self.send_job(job1)
        job2 = czf_pb2.Job(
            model=self.model_2p,
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_2P],
            step=0,
            workers=self._workers,
        )
        await self.send_job(job2)
