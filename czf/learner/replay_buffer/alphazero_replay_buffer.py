'''CZF AlphaZero Replay Buffer'''
from collections import deque, namedtuple
from itertools import zip_longest

from czf.learner.replay_buffer.replay_buffer import ReplayBuffer

import random
import numpy as np
import czf.env.czf_env

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
        use_transformation,
        game,
    ):
        super().__init__(
            num_player,
            states_to_train,
            sequences_to_train,
            sample_ratio,
            sample_states,
            capacity,
        )
        self._buffer = deque(maxlen=capacity)
        self._use_transformation = use_transformation
        self._game = czf.env.czf_env.load_game(game)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        transition = self._buffer[index]

        to_ndarray = lambda x, dtype=np.float32: np.array(np.frombuffer(x, dtype=dtype))
        to_bytes = lambda x, dtype=np.float32: np.array(x, dtype=dtype).tobytes()

        if self._use_transformation:
            # transformation type
            transformation_type = random.randint(0, self._game.num_transformations - 1)
            # observation
            observation_list = self._game.transform_observation(to_ndarray(transition.observation),
                                                                transformation_type)
            transform_observation = to_bytes(observation_list)
            # policy
            policy_list = self._game.transform_policy(to_ndarray(transition.policy),
                                                      transformation_type)
            transform_policy = to_bytes(policy_list)
            # return
            return [
                transform_observation,
                transform_policy,
                transition.value,
            ]
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
