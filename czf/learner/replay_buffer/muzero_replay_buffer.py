'''CZF MuZero Replay Buffer'''
from collections import deque, namedtuple
from itertools import zip_longest
import numpy as np
import zstandard as zstd

from czf.learner.replay_buffer.replay_buffer import ReplayBuffer

MuZeroTransition = namedtuple('MuZeroTransition', [
    'observation',
    'action',
    'policy',
    'value',
    'reward',
    'is_terminal',
    'valid',
])


class MuZeroRolloutBatch:
    '''MuZeroRolloutBatch is used to collate data samples from :class:`ReplayBuffer`.

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


class MuZeroTransitionBuffer:
    '''MuZeroTransitionBuffer is mainly used to stack frames.
    When `frame_stack` equals to 0, it essentially works as a deque with compression.

    Usage:

    - Store transitions with compression
    - Get a transition with stacked features from the buffer
    '''
    def __init__(self, maxlen, frame_stack, spatial_shape):
        self._weights_mean = 0.
        self._weights_temp = None
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
        self._weights_mean = np.mean([w for w in weights if w > 0.])
        return weights

    def get_mean_weight(self):
        '''Get average weights of prioritized replay'''
        return self._weights_mean

    def update_weights(self, index, priorities):
        '''Update weights'''
        for i, priority in zip(index, priorities):
            self._weights_temp[i + self._frame_stack] = (priority + 1e-5)

    def copy_weights(self):
        '''Copy weights for further updates'''
        self._weights_temp = self._weights.copy()

    def write_back_weights(self):
        '''Write back updated weights'''
        self._weights = self._weights_temp.copy()

    def get_valid_index(self, index):
        '''Get a valid index'''
        transition = self.__getitem__(index)
        assert transition.valid
        # self._frame_stack + self._mstep
        return index

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
        # concat feature tensors into an observation
        # (a_1, o_1), (a_2, o_2), ..., (a_n, o_n)
        return weight, b''.join([
            self.__getitem__(index - i).observation
            for i in reversed(range(self._frame_stack))
        ])

    def extend(self, priorities, trajectories):
        '''Extend the right side of the buffer by
        appending elements from the iterable argument.'''
        self._weights.extend(priorities)
        self._buffer.extend(trajectories)


class MuZeroReplayBuffer(ReplayBuffer):
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
    def __init__(
        self,
        num_player,
        states_to_train,
        sequences_to_train,
        sample_ratio,
        sample_states,
        observation_config,
        capacity,
        kstep,
    ):
        super().__init__(
            num_player,
            states_to_train,
            sequences_to_train,
            sample_ratio,
            sample_states,
        )
        self._spatial_shape = observation_config['spatial_shape']
        self._frame_stack = observation_config['frame_stack']
        self._kstep = kstep
        self._buffer = MuZeroTransitionBuffer(capacity, self._frame_stack,
                                              self._spatial_shape)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        index = self._buffer.get_valid_index(index)
        weight, observation = self._buffer.get_observation(index)
        rollout = self._buffer.get_rollout(index, self._kstep + 1)
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
        kstep -= 1
        scale = np.array(1. / kstep, dtype=np.float32).tobytes()
        # i, w, o, s, [(v, m, p, a, r)]
        return [
            index,
            weight,
            observation,
            scale,
            *transitions,
        ]

    def _extend(self, priorities, trajectories):
        self._buffer.extend(priorities, trajectories)

    def get_weights(self):
        '''Get weights (not summing up to one) of samples'''
        return self._buffer.get_weights()

    def get_mean_weight(self):
        '''Get average weights of prioritized replay'''
        return self._buffer.get_mean_weight()

    def update_weights(self, index, priorities):
        '''Update weights'''
        self._buffer.update_weights(index, priorities)

    def copy_weights(self):
        '''Copy weights for further updates'''
        self._buffer.copy_weights()

    def write_back_weights(self):
        '''Write back updated weights'''
        self._buffer.write_back_weights()
