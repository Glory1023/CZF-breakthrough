'''CZF Dataloader'''
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import zip_longest
import numpy as np
import torch
import torch.multiprocessing as mp
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
Transition.__doc__ = '''Transition is used to store in :class:`TransitionBuffer`'''


@dataclass
class Statistics:
    '''Statistics for a :class:`ReplayBuffer`

    :param num_games: total number of games
    :param game_steps: a list of steps for each game
    :param player_returns: returns for each player
    '''
    num_games: int
    game_steps: list
    player_returns: list


class RolloutBatch:
    '''RolloutBatch is used to collate data samples from :class:`ReplayBuffer`.

    Each data sample is a list of `torch.Tensor` and may (intentionally)
    contains serveral `None` at the end. As a result, we not only have to
    stack `torch.Tensor`, but we also have to discard `None`, which is the
    `sentinel` of the list.
    '''
    def __init__(self, data):
        index, weight, observation, scale, *transition = self.__zip_discard(
            *data)
        self.index = tuple(index)
        self.weight = b''.join(weight)
        self.observation = b''.join(observation)
        self.scale = b''.join(scale)
        self.transition = tuple(b''.join(d) for d in transition if d)

    @staticmethod
    def __zip_discard(*iterables, sentinel=None):
        '''zip longest and discard `sentinel`'''
        return [[entry for entry in iterable if entry is not sentinel]
                for iterable in zip_longest(*iterables, fillvalue=sentinel)]


class TransitionBuffer:
    '''TransitionBuffer is mainly used to stack frames.
    When `frame_stack` equals to 0, it essentially works as a deque with compression.

    Usage:

    - Store transitions with compression
    - Get a transition with stacked features from the buffer
    '''
    def __init__(self, maxlen, frame_stack, spatial_shape):
        self._weights_mean = 0.
        self._weights = deque(maxlen=(maxlen + frame_stack))
        self._buffer = deque(maxlen=(maxlen + frame_stack))
        self._frame_stack = frame_stack
        self._spatial_shape = spatial_shape
        # self._dctx = zstd.ZstdDecompressor()

    def __len__(self):
        return len(self._buffer) - self._frame_stack

    def __getitem__(self, index):
        return self._buffer[index + self._frame_stack]

    def get_weights(self):
        '''Get weights (not summing up to one) of samples'''
        weights = list(self._weights)[self._frame_stack:]
        self._weights_mean = np.mean(weights)
        return weights

    def update_weights(self, index, priorities):
        '''Update weights'''
        for i, priority in zip(index, priorities):
            self._weights_temp[i + self._frame_stack] = (priority + 1e-6)

    def copy_weights(self):
        '''Copy weights for further updates'''
        self._weights_temp = self._weights.copy()

    def write_back_weights(self):
        '''Write back updated weights'''
        self._weights = self._weights_temp.copy()

    def get_rollout(self, index, kstep):
        '''Get rollout at index with kstep'''
        buffer_len = len(self)
        return [
            self.__getitem__(index + i) for i in range(kstep)
            if index + i < buffer_len
        ]

    def get_observation(self, index):
        '''Get observation (stacked features) at index'''
        weight = self._weights_mean / self._weights[index + self._frame_stack]
        weight = np.array(weight, dtype=np.float32).tobytes()
        if self._frame_stack == 0:
            return weight, self.__getitem__(index).observation
        buffer, first_index = [], index
        for i in range(self._frame_stack):
            transition = self.__getitem__(index - i)
            if transition.is_terminal:
                first_index = index - i + 1
                break
            buffer.append(transition.observation)
        if len(buffer) < self._frame_stack:
            # repeat initial observation
            buffer.extend([
                self.__getitem__(first_index).observation
                for _ in range(self._frame_stack - len(buffer))
            ])
        # concat feature tensors into an observation
        # (a_1, o_1), (a_2, o_2), ..., (a_n, o_n)
        return weight, b''.join(reversed(buffer))

    def extend(self, priorities, trajectory):
        '''Extend the right side of the buffer by
        appending elements from the iterable argument.'''
        self._weights.extend(priorities)
        self._buffer.extend(trajectory)


class Preprocessor:
    '''Preprocess raw packet to trajectories'''
    def __init__(
        self,
        result_queue,
        num_player,
        num_action,
        observation_config,
        transform,
        r_heads,
        v_heads,
        kstep,
        nstep,
        discount_factor,
    ):
        self._result_queue = result_queue
        self._spatial_shape = observation_config['spatial_shape']
        self._frame_stack = observation_config['frame_stack']
        self._num_action = num_action
        self._num_player = num_player
        self._transform = None
        if transform == 'Atari':
            # sign(x) * (sqrt(|x| + 1) − 1 + eps * x)
            epsilon = 0.001
            self._transform = lambda x: np.sign(x) * (np.sqrt(np.abs(x) + 1) -
                                                      1 + epsilon * x)
        self._r_heads = r_heads
        self._v_heads = v_heads
        self._kstep = kstep
        self._nstep = nstep
        self._discount_factor = discount_factor

    @staticmethod
    def __get_target_dist(x, heads):
        '''Returns the target distribution of a reward or a value'''
        x_low, x_high = int(np.floor(x)), int(np.ceil(x))
        target = np.zeros(2 * heads + 1, dtype=np.float32)
        target[x_low + heads] = x_high - x
        target[x_high + heads] = 1 - target[x_low + heads]
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

        #print('add', len(trajectory.states), 'positions')
        # TODO: toggle save trajectory (may exceed 2G!)
        # self._pb_trajectory_batch.trajectories.add().CopyFrom(trajectory)
        # from the terminal state to the initial state
        nstep, gamma = self._nstep, self._discount_factor
        discounted_return = [[] for _ in range(self._num_player)]
        values = []
        priorities, buffer = [], []
        has_statistics = trajectory.HasField('statistics')
        for i, state in enumerate(reversed(trajectory.states)):
            # tensor
            observation = to_bytes(state.observation_tensor)
            # observation = self._cctx_observation.compress(observation)
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
                    terminal_value = state.transition.rewards[
                        state.transition.current_player]
                if self._transform is not None:
                    terminal_value = self._transform(terminal_value)
                    terminal_value = self.__get_target_dist(
                        terminal_value, self._v_heads)
                else:
                    terminal_value = [terminal_value]
                terminal_value = to_bytes(terminal_value)
                priorities.append(0.)
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
            reward = state.transition.rewards[state.transition.current_player]
            # priority
            priority = 1.
            if self._num_player == 1:
                # alpha == 1
                priority = np.abs(state.evaluation.value - nstep_value)
            priorities.append(priority + 1e-6)
            # transform
            if self._transform is not None:
                nstep_value = self._transform(nstep_value)
                nstep_value = self.__get_target_dist(nstep_value,
                                                     self._v_heads)
                reward = self._transform(reward)
                reward = self.__get_target_dist(reward, self._r_heads)
            else:
                nstep_value = [nstep_value]
                reward = [reward]
            # tensor
            action = to_bytes(state.transition.action)
            policy = to_bytes(state.evaluation.policy)
            value = to_bytes(nstep_value)
            reward = to_bytes(reward)
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
            del state
        # update statistics
        stats = Statistics(
            num_games=1,
            game_steps=[len(trajectory.states) - 1],
            player_returns=[[] for _ in range(self._num_player)],
        )
        if has_statistics:
            stats.game_steps[:] = [trajectory.statistics.game_steps]
            for player, reward in enumerate(trajectory.statistics.rewards):
                stats.player_returns[player].append(reward)
        # add trajectory to buffer (from start to terminal)
        self._result_queue.put(
            (stats, list(reversed(priorities)), list(reversed(buffer))))


def run_preprocessor(raw_queue, *args):
    '''run :class:`EnvManager`'''
    preprocessor = Preprocessor(*args)
    while True:
        raw = raw_queue.get()
        packet = czf_pb2.Packet.FromString(raw)
        batch = packet.trajectory_batch.trajectories
        for trajectory in batch:
            preprocessor.add_trajectory(trajectory)
        packet.ClearField('trajectory_batch')


class PreprocessQueue:
    '''Preprocess job queue'''
    def __init__(
        self,
        num_player,
        num_action,
        observation_config,
        transform,
        r_heads,
        v_heads,
        kstep,
        nstep,
        discount_factor,
        num_proc,
    ):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # multiprocessing
        self._raw_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._process = [
            mp.Process(target=run_preprocessor,
                       args=(
                           self._raw_queue,
                           self._result_queue,
                           num_player,
                           num_action,
                           observation_config,
                           transform,
                           r_heads,
                           v_heads,
                           kstep,
                           nstep,
                           discount_factor,
                       )) for _ in range(num_proc)
        ]
        for process in self._process:
            process.start()

    def put(self, raw: bytes):
        '''Enqueue raw protobuf packet'''
        self._raw_queue.put(raw)

    def get(self):
        '''Dequeue preprocessed trajectory'''
        return self._result_queue.get()


class ReplayBuffer(Dataset):
    '''ReplayBuffer is used to store and sample transitions.

    * | `__getitem__` support fetching a data sample for a given index.
      | If the data sample is a terminal state, then the index is shifted
      | in one element forward or backward. A data sample is a list of
      | `torch.Tensor` and may contains several `None` at the end due to terminals.
      | In addition, a data sample contains the following:

        .. code-block:: python

            [
                observation,                     # a stacked feature
                gradient_scale,                  # 1 / K
                *[value, mask, policy, reward],  # K-steps transitions
            ]

      | For example, if the next state of `obs` is terminal, then the data sample is:

        .. code-block:: python

            [
                obs,
                1.,              # = 1 / 1
                terminal_value,  # `value`: the value of terminal state
                0.,              # `mask`: terminal state has no next state
                None,            # `policy`: terminal state has no policy
                None,            # `reward`: terminal state has no reward
            ]
    '''
    def __init__(self, num_player, observation_config, kstep, capacity,
                 train_freq):
        self._spatial_shape = observation_config['spatial_shape']
        self._frame_stack = observation_config['frame_stack']
        self._num_player = num_player
        self._kstep = kstep
        self._train_freq = train_freq
        self._buffer = TransitionBuffer(capacity, self._frame_stack,
                                        self._spatial_shape)
        self._pb_trajectory_batch = czf_pb2.TrajectoryBatch()
        # self._cctx_observation = zstd.ZstdCompressor()
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
        weight, observation = self._buffer.get_observation(index)
        rollout = self._buffer.get_rollout(index, self._kstep)
        kstep, transitions = 0, []
        np0 = np.array(0, dtype=np.float32).tobytes()
        np1 = np.array(1, dtype=np.float32).tobytes()
        for transition in rollout:
            kstep += 1
            transitions.extend([
                transition.value,
                np0 if transition.is_terminal else np1,
                transition.policy,
                transition.action,
                transition.reward,
            ])
            if transition.is_terminal:
                break
        scale = np.array(1. / kstep, dtype=np.float32).tobytes()
        # i, w, o, s, [(v, m, p, a, r)]
        return [
            index,
            weight,
            observation,
            scale,
            *transitions,
        ]

    def get_weights(self):
        '''Get weights (not summing up to one) of samples'''
        return self._buffer.get_weights()

    def update_weights(self, index, priorities):
        '''Update weights'''
        self._buffer.update_weights(index, priorities)

    def copy_weights(self):
        '''Copy weights for further updates'''
        self._buffer.copy_weights()

    def write_back_weights(self):
        '''Write back updated weights'''
        self._buffer.write_back_weights()

    def is_ready(self):
        '''Check if replay buffer is ready for training'''
        if self._ready:
            self._ready = False
            return True
        return False

    def add_trajectory(self, trajectory: tuple):
        '''Add a trajectory and its statistics'''
        stats, priorities, trajectory = trajectory
        self._num_games += sum(stats.game_steps)
        # update statistics
        self._statistics.num_games += stats.num_games
        self._statistics.game_steps.extend(stats.game_steps)
        for player, player_returns in enumerate(stats.player_returns):
            self._statistics.player_returns[player].extend(player_returns)
        # add trajectory to buffer
        self._buffer.extend(priorities, trajectory)
        # train the model when there are N newly generated states
        if self._num_games >= self._train_freq:
            self._ready = True
            self._num_games -= self._train_freq

    def get_statistics(self):
        '''Returns :class:`Statistics` of recent trajectories'''
        return self._statistics

    def reset_statistics(self):
        '''Reset the :class:`Statistics` information of recent trajectories'''
        self._statistics = Statistics(
            num_games=0,
            game_steps=[],
            player_returns=[[] for _ in range(self._num_player)],
        )

    def save_trajectory(self, path, iteration):
        '''Save all trajectories to the `path` with compression,
        and clear up all trajactories'''
        trajectory = self._pb_trajectory_batch.SerializeToString()
        compressed = self._cctx_trajectory.compress(trajectory)
        trajectory_path = path / f'{iteration:05d}.pb.zst'
        trajectory_path.write_bytes(compressed)
        self._pb_trajectory_batch.Clear()
