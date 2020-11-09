'''CZF Evaluator'''
import asyncio
import numpy as np

from czf.game_server.game_server import GameServer, EnvManager
from czf.pb import czf_pb2


class EvalEnvManager(EnvManager):
    '''Game Evaluation Environment Manager'''
    async def send_search_job(self):
        '''helper to send a `Job` to actor'''
        operation = self.operation[self._state.current_player]
        job = czf_pb2.Job(
            procedure=[operation],
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
        job.model.CopyFrom(self._server.models[operation])
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
            first_player = self.operation[0]
            if first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_1P:
                reward = self._state.rewards[0]
            else:  # first_player == czf_pb2.Job.Operation.MUZERO_EVALUATE_2P:
                reward = self._state.rewards[1]
            self.reset()
            return True, reward
        return False, None


class EvalGameServer(GameServer):
    '''Evalutation Game Server'''
    def __init__(self, args, config, callbacks):
        super().__init__(args, config, callbacks)
        eval_config = config['evaluator']
        assert eval_config['mode'] == 'best'
        self._freq = eval_config['frequency']
        num_env = args.num_env
        self._total = 2 * num_env
        self._workers = [czf_pb2.Node()]
        # 2 * num_env (1P/2P exchange)
        self.envs = [
            EvalEnvManager(self, start=False) for _ in range(self._total)
        ]
        for i in range(num_env):
            self.envs[i].operation = [
                czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
            ]
            self.envs[num_env + i].operation = [
                czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
                czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
            ]
        # in best mode, 1p is the latest, 2p is the best
        self._model_1p = czf_pb2.ModelInfo(name='default', version=0)
        self._model_2p = czf_pb2.ModelInfo(name='default', version=0)
        self.models = {
            czf_pb2.Job.Operation.MUZERO_EVALUATE_1P: self._model_1p,
            czf_pb2.Job.Operation.MUZERO_EVALUATE_2P: self._model_2p,
        }
        self.has_new_model.set()
        self._best_elo = 0.
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
                self._model_1p.version,
                self._model_2p.version,
                win_rate,
                draw_rate,
                lose_rate,
                elo_1p,
                elo_1p_diff,
            ))
        # send the evaluation result
        result = czf_pb2.EvaluationResult(
            iteration=self._model_1p.version,
            elo=elo_1p,
            win=win_rate,
            draw=draw_rate,
            lose=lose_rate,
        )
        result.target.CopyFrom(self._model_1p)
        result.base.CopyFrom(self._model_2p)
        packet = czf_pb2.Packet(evaluation_result=result)
        await self._upstream.send(packet.SerializeToString())
        # update model according to the score
        if score > 0.55:  # current model (1p) is the best
            self._model_2p.CopyFrom(self._model_1p)
            self._best_elo = elo_1p
        await self.__send_load_model()

    async def __send_load_model(self):
        '''helper to wait for new model and send jobs to flush'''
        # wait for next model version
        next_version = self._model_1p.version + self._freq
        if self.model.version < next_version:
            while await self.has_new_model.wait():
                self.has_new_model.clear()
                if self.model.version >= next_version:
                    self._model_1p.version = next_version
                    break
        else:
            self._model_1p.version = next_version
        # send jobs to flush model
        job1 = czf_pb2.Job(
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_1P],
            step=0,
            workers=self._workers,
        )
        job1.model.CopyFrom(self._model_1p)
        await self.send_job(job1)
        job2 = czf_pb2.Job(
            procedure=[czf_pb2.Job.Operation.MUZERO_EVALUATE_2P],
            step=0,
            workers=self._workers,
        )
        job2.model.CopyFrom(self._model_2p)
        await self.send_job(job2)
