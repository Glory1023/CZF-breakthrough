'''CZF MuZero Preprocessor'''
import numpy as np
import torch.multiprocessing as mp

from czf.learner.preprocessor.preprocessor import Preprocessor
from czf.learner.replay_buffer.replay_buffer import Statistics
from czf.learner.replay_buffer.muzero_replay_buffer import MuZeroTransition
from czf.pb import czf_pb2


class MuZeroPreprocessor(Preprocessor):
    '''MuZero Preprocess raw packet to trajectories'''
    def __init__(
        self,
        result_queue,
        num_player,
        transform,
        r_heads,
        v_heads,
        kstep,
        nstep,
        discount_factor,
        use_prioritize,
    ):
        self._result_queue = result_queue
        self._num_player = num_player
        self._transform = None
        if transform == 'Atari':
            # sign(x) * (sqrt(|x| + 1) − 1 + eps * x)
            epsilon = 0.001
            self._transform = lambda x: np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + epsilon * x)
        self._r_heads = r_heads
        self._v_heads = v_heads
        self._kstep = kstep
        self._nstep = nstep
        self._mstep = kstep + nstep
        self._discount_factor = discount_factor
        self._use_prioritize = use_prioritize

    @staticmethod
    def _get_target_dist(x, heads):
        '''Returns the target distribution of a reward or a value'''
        h_low, h_high = heads
        x = np.clip(x, h_low, h_high)
        x_low, x_high = int(np.floor(x)), int(np.ceil(x))
        target = np.zeros(h_high - h_low + 1, dtype=np.float32)
        target[x_low + h_low] = x_high - x
        target[x_high + h_low] = 1 - target[x_low + h_low]
        return target

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
        # from the terminal state to the initial state
        nstep, gamma = self._nstep, self._discount_factor
        discounted_return = [[] for _ in range(self._num_player)]
        values = []
        priorities, buffer_ = [], []
        has_statistics = trajectory.HasField('statistics')
        num_states = 0
        for i, state in enumerate(reversed(trajectory.states)):
            # tensor
            observation = to_bytes(state.observation_tensor)
            if i == 0:
                # terminal returns and values (equal to 0 if _real_ terminal)
                terminal_next_value = 0. if has_statistics else state.evaluation.value
                for player in range(self._num_player):
                    discounted_return[player].append(terminal_next_value)
                values.append(terminal_next_value)
                # terminal transition
                if self._num_player == 1:
                    terminal_value = terminal_next_value
                else:
                    terminal_value = state.transition.rewards[state.transition.current_player]
                if self._transform is not None:
                    terminal_value = self._transform(terminal_value)
                    terminal_value = self._get_target_dist(
                        terminal_value,
                        self._v_heads,
                    )
                else:
                    terminal_value = [terminal_value]
                terminal_value = to_bytes(terminal_value)
                priorities.append(0.)
                buffer_.append(
                    MuZeroTransition(
                        observation=observation,
                        action=None,
                        policy=None,
                        value=terminal_value,
                        reward=None,
                        is_terminal=True,
                        valid=False,
                    ))
                continue
            if not state.HasField('transition'):  # is_frame_stack
                priorities.append(0.)
                buffer_.append(
                    MuZeroTransition(
                        observation=observation,
                        action=None,
                        policy=None,
                        value=None,
                        reward=None,
                        is_terminal=False,
                        valid=False,
                    ))
                continue
            is_valid = has_statistics or (i >= self._mstep)
            # monte carlo returns
            for player in range(self._num_player):
                rewards = state.transition.rewards
                discounted_return[player].append(rewards[player] +
                                                 gamma * discounted_return[player][-1])
            values.append(state.evaluation.value)
            # nstep value
            player = state.transition.current_player
            nstep_value = discounted_return[player][-1]
            if (nstep > 0) and (i >= nstep):
                nstep_value += (values[-nstep - 1] -
                                discounted_return[player][-nstep - 1]) * gamma**nstep
            reward = state.transition.rewards[state.transition.current_player]
            # priority
            priority = 1.
            if self._num_player == 1 and self._use_prioritize:
                # alpha == 1
                priority = np.abs(state.evaluation.value - nstep_value)
            if is_valid:
                priorities.append(priority + 1e-5)
                num_states += 1
            else:
                priorities.append(0.)
            # transform
            if self._transform is not None:
                nstep_value = self._transform(nstep_value)
                nstep_value = self._get_target_dist(
                    nstep_value,
                    self._v_heads,
                )
                reward = self._transform(reward)
                reward = self._get_target_dist(reward, self._r_heads)
            else:
                nstep_value = [nstep_value]
                reward = [reward]
            # tensor
            action = to_bytes(state.transition.action)
            policy = to_bytes(state.evaluation.policy)
            value = to_bytes(nstep_value)
            reward = to_bytes(reward)
            # (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
            buffer_.append(
                MuZeroTransition(
                    observation=observation,
                    action=action,
                    policy=policy,
                    value=value,
                    reward=reward,
                    is_terminal=False,
                    valid=is_valid,
                ))
            del state
        # update statistics
        stats = Statistics(
            num_games=1,
            num_states=num_states,
            game_steps=tuple(),
            player_returns=tuple(tuple() for _ in range(self._num_player)),
        )
        if has_statistics:
            stats = Statistics(
                num_games=1,
                num_states=num_states,
                game_steps=(trajectory.statistics.game_steps, ),
                player_returns=tuple((reward, ) for reward in trajectory.statistics.rewards),
            )
        # add trajectory to buffer (from start to terminal)
        self._result_queue.put((stats, tuple(reversed(priorities)), tuple(reversed(buffer_))))
