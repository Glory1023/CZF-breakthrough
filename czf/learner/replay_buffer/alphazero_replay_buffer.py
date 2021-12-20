'''CZF AlphaZero Replay Buffer'''
from collections import deque, namedtuple
from itertools import zip_longest

from czf.learner.replay_buffer.replay_buffer import ReplayBuffer

AlphaZeroTransition = namedtuple('AlphaZeroTransition', [
    'observation',
    'policy',
    'value',
])


class AlphaZeroBatch:
    def __init__(self, data):
        observation, policy, value = self.__zip_discard(*data)
        self.observation = b''.join(observation)
        self.policy = b''.join(policy)
        self.value = b''.join(value)

    @staticmethod
    def __zip_discard(*iterables, sentinel=None):
        '''zip longest and discard `sentinel`'''
        return [[entry for entry in iterable if entry is not sentinel]
                for iterable in zip_longest(*iterables, fillvalue=sentinel)]


class AlphaZeroReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        num_player,
        states_to_train,
        sequences_to_train,
        sample_ratio,
        sample_states,
        observation_config,
        capacity,
    ):
        super().__init__(
            num_player,
            states_to_train,
            sequences_to_train,
            sample_ratio,
            sample_states,
        )
        self._buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        transition = self._buffer[index]
        return [
            transition.observation,
            transition.policy,
            transition.value,
        ]

    def _extend(self, priorities, trajectories):
        self._buffer.extend(trajectories)

    def get_weights(self):
        '''Get weights (not summing up to one) of samples'''
        weights = [1.] * len(self._buffer)
        return weights
