'''CZF AlphaZero Preprocessor'''
import numpy as np
import torch.multiprocessing as mp

from czf.learner.preprocessor.preprocessor import Preprocessor
from czf.learner.replay_buffer.replay_buffer import Statistics
from czf.learner.replay_buffer.alphazero_replay_buffer import AlphaZeroTransition
from czf.pb import czf_pb2


class AlphaZeroPreprocessor(Preprocessor):
    '''AlphaZero Preprocess raw packet to trajectories'''
    def __init__(self, result_queue):
        self._result_queue = result_queue

    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''Add a trajectory to the replay buffer

        - | Process the trajectory into a sequence of :class:`Transition`:
          | (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
        - Update the :class:`Statistics` information of the replay buffer.
        - | Check if the replay buffer is ready to update.
          | That is, current number of games is not less than `train_freq`
        '''
        def to_bytes(x, dtype=np.float32):
            x = np.array(x, dtype=dtype)
            return x.tobytes()

        # print('add', len(trajectory.states), 'positions')
        # TODO: toggle save trajectory (may exceed 2G!)
        # self._pb_trajectory_batch.trajectories.add().CopyFrom(trajectory)
        # from the initial state to the terminal state
        num_states = len(trajectory.states)
        priorities = [1.] * num_states
        buffer = []
        for state in trajectory.states:
            # tensor
            observation = to_bytes(state.observation_tensor)
            policy = to_bytes(state.evaluation.policy)
            returns = to_bytes(trajectory.statistics.rewards)
            # (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
            buffer.append(
                AlphaZeroTransition(
                    observation=observation,
                    policy=policy,
                    value=returns,
                ))
            del state
        # update statistics
        stats = Statistics(
            num_games=1,
            num_states=num_states,
            game_steps=(trajectory.statistics.game_steps, ),
            player_returns=tuple(
                (reward, ) for reward in trajectory.statistics.rewards),
        )
        # add trajectory to buffer (from start to terminal)
        self._result_queue.put(
            (stats, tuple(priorities), tuple(buffer)))
