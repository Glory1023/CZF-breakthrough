'''CZF Dataloader'''
from collections import Counter, deque, namedtuple
from dataclasses import dataclass
from itertools import islice, zip_longest
import numpy as np
import torch
from torch.utils.data import Dataset
import zstandard as zstd

from czf.pb import czf_pb2

Transition = namedtuple('Transition', [
    'observation',
    'action',
    'policy',
    'value',
    'reward',
    'is_terminal',
])


@dataclass
class Statistics:
    '''Statistics for a `ReplayBuffer`

    * `num_games`: total number of games
    * `game_steps`: a list of steps for each game
    * `player_returns`: returns for each player
    '''
    num_games: int
    game_steps: list
    player_returns: list


class RolloutBatch:
    '''RolloutBatch is used to collate data samples from `ReplayBuffer`.

    Each data sample is a list of `torch.Tensor` and may (intentionally)
    contains serveral `None` at the end. As a result, we not only have to
    stack `torch.Tensor`, but we also have to discard `None`, which is the
    `sentinel` of the list.
    '''
    def __init__(self, data):
        observation, scale, *transition = self.__zip_discard(*data)
        self.observation = torch.stack(observation, 0)
        self.scale = torch.stack(scale, 0)
        self.transition = [torch.stack(d, 0) for d in transition if d]

    @staticmethod
    def __zip_discard(*iterables, sentinel=None):
        '''zip longest and discard `sentinel`'''
        return [[entry for entry in iterable if entry is not sentinel]
                for iterable in zip_longest(*iterables, fillvalue=sentinel)]


class TransitionBuffer:
    '''TransitionBuffer is mainly used to stack frames.
    When `frame_stack` equals to 0, it essentially works as a deque with compression.

    Usage:
    * Store transitions with compression
    * Get a transition with stacked features from the buffer
    '''
    def __init__(self, maxlen, frame_stack, spatial_shape):
        self._buffer = deque(maxlen=(maxlen + frame_stack))
        self._frame_stack = frame_stack
        self._spatial_shape = spatial_shape
        self._dctx = zstd.ZstdDecompressor()

    def __len__(self):
        return len(self._buffer) - self._frame_stack

    def __getitem__(self, index):
        return self._buffer[index + self._frame_stack]

    def get_rollout(self, index, kstep):
        '''Get rollout at index with kstep'''
        index += self._frame_stack
        return list(islice(self._buffer, index, index + kstep))

    def get_observation(self, index):
        '''Get observation (stacked features) at index.'''
        def to_tensor(obs):
            obs = self._dctx.decompress(obs)
            obs = np.array(np.frombuffer(obs, dtype=np.float32))
            return torch.tensor(obs).view(-1, *self._spatial_shape)

        if self._frame_stack == 0:
            return to_tensor(self._buffer[index].observation)
        # concat feature tensors into an observation
        # (a_1, o_1), (a_2, o_2), ..., (a_n, o_n)
        return torch.cat([
            to_tensor(self._buffer[index + i].observation)
            for i in range(self._frame_stack)
        ])

    def extend(self, data):
        '''Extend the right side of the buffer by
        appending elements from the iterable argument.'''
        self._buffer.extend(data)


class ReplayBuffer(Dataset):
    '''ReplayBuffer is used to store and sample transitions.

    * `__getitem__` support fetching a data sample for a given index.
      If the data sample is a terminal state, then the index is shifted
      in one element forward or backward. A data sample is a list of
      `torch.Tensor` and may contains several `None` at the end due to
      terminals.

      In addition, a data sample contains the following: [
          observation,                     # a stacked feature
          gradient_scale,                  # 1 / K
          *[value, mask, policy, reward],  # K-steps transitions
      ]
      For example, if the next state of `obs` is terminal, then
      the data sample is: [
          `obs`,
          1.,              # = 1 / 1
          terminal_value,  # `value`: the value of terminal state
          0.,              # `mask`: terminal state has no next state
          None,            # `policy`: terminal state has no policy
          None,            # `reward`: terminal state has no reward
      ]
    '''
    def __init__(self, num_player, num_action, observation_config, kstep,
                 nstep, discount_factor, capacity, train_freq):
        self._spatial_shape = observation_config['spatial_shape']
        self._frame_stack = observation_config['frame_stack']
        self._num_action = num_action
        self._num_player = num_player
        self._kstep = kstep
        self._nstep = nstep
        self._discount_factor = discount_factor
        self._train_freq = train_freq
        self._buffer = TransitionBuffer(capacity, self._frame_stack,
                                        self._spatial_shape)
        self._pb_trajectory_batch = czf_pb2.TrajectoryBatch()
        self._cctx_observation = zstd.ZstdCompressor()
        self._cctx_trajectory = zstd.ZstdCompressor()
        self._num_games = 0
        self._ready = False
        self.reset_statistics()

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        if self._buffer[index].is_terminal:
            if index == 0:
                index += 1
            else:
                index -= 1
        observation = self._buffer.get_observation(index)
        rollout = self._buffer.get_rollout(index, self._kstep)
        kstep, transitions = 0, []
        for transition in rollout:
            kstep += 1
            transitions.extend([
                transition.value,
                torch.tensor(0 if transition.is_terminal else 1),
                transition.policy,
                transition.action,
                transition.reward,
            ])
            if transition.is_terminal:
                break
        # o, s, [(v, m, p, a, r)]
        return [observation, torch.tensor(1 / kstep), *transitions]

    def is_ready(self):
        '''Check if replay buffer is ready for training.'''
        if self._ready:
            self._ready = False
            return True
        return False

    def _stack_action_observation(self, action, obs):
        '''Returns a feature of stacked action and observation.'''
        normalized_action = (action + 1.) / self._num_action
        action_plane = np.full(np.prod(self._spatial_shape),
                               normalized_action,
                               dtype=np.float32)
        return np.hstack((action_plane, obs))

    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''Add a trajectory to the replay buffer

        * Process the trajectory into a sequence of `Transition`:
          (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
        * Update the `Statistics` information of the replay buffer.
        * Check if the replay buffer is ready to update.
          That is, current number of games is not less than `train_freq`
        '''
        #print('add', len(trajectory.states), 'positions')
        self._pb_trajectory_batch.trajectories.add().CopyFrom(trajectory)
        self._num_games += len(trajectory.states)
        self._statistics.num_games += 1
        self._statistics.game_steps.append(len(trajectory.states))
        # from the terminal state to the initial state
        nstep, gamma = self._nstep, self._discount_factor
        discounted_return = [[] for _ in range(self._num_player)]
        values = []
        buffer = []
        reversed_states = list(reversed(trajectory.states))
        for i, state in enumerate(reversed_states):
            # tensor
            observation = np.array(state.observation_tensor, dtype=np.float32)
            if self._frame_stack > 0:
                prev_action = reversed_states[i + 1].transition.action if (
                    i + 1 < len(reversed_states)) else -1
                observation = self._stack_action_observation(
                    prev_action, observation)
            observation = observation.tobytes()
            observation = self._cctx_observation.compress(observation)
            if i == 0:
                # update statistics
                for player in range(self._num_player):
                    self._statistics.player_returns[player].update(
                        [str(state.transition.rewards[player])])
                # terminal returns and values (equal to 0 if _real_ terminal)
                for player in range(self._num_player):
                    discounted_return[player].append(state.evaluation.value)
                values.append(state.evaluation.value)
                # terminal transition
                terminal_value = torch.tensor([
                    state.transition.rewards[state.transition.current_player]
                ])
                buffer.append(
                    Transition(
                        observation=observation,
                        action=None,
                        policy=None,
                        value=terminal_value,
                        reward=None,
                        is_terminal=True,
                    ))
                continue
            # monte carlo returns
            for player in range(self._num_player):
                rewards = state.transition.rewards
                discounted_return[player].append(rewards[player] + gamma *
                                                 discounted_return[player][-1])
            values.append(state.evaluation.value)
            # nstep value
            player = state.transition.current_player
            nstep_value = discounted_return[player][-1]
            if (nstep > 0) and (i >= nstep):
                nstep_value += (
                    values[-nstep] -
                    discounted_return[player][-nstep - 1]) * gamma**nstep
            # tensor
            action = torch.tensor(state.transition.action)
            policy = torch.tensor(state.evaluation.policy)
            value = torch.tensor([nstep_value])
            reward = torch.tensor(
                [state.transition.rewards[state.transition.current_player]])
            # (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
            buffer.append(
                Transition(
                    observation=observation,
                    action=action,
                    policy=policy,
                    value=value,
                    reward=reward,
                    is_terminal=False,
                ))
        self._buffer.extend(reversed(buffer))

        if self._num_games >= self._train_freq:
            self._ready = True
            self._num_games -= self._train_freq

    def get_statistics(self):
        '''Returns `Statistics` of recent trajectories'''
        return self._statistics

    def reset_statistics(self):
        '''Reset the `Statistics` information of recent trajectories'''
        self._statistics = Statistics(
            num_games=0,
            game_steps=[],
            player_returns=[Counter() for _ in range(self._num_player)],
        )

    def save_trajectory(self, path, iteration):
        '''Save all trajectories to the `path` with compression,
        and clear up all trajactories'''
        trajectory = self._pb_trajectory_batch.SerializeToString()
        compressed = self._cctx_trajectory.compress(trajectory)
        trajectory_path = path / f'{iteration:05d}.pb.zst'
        trajectory_path.write_bytes(compressed)
        self._pb_trajectory_batch.Clear()
