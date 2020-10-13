'''CZF Dataloader'''
from collections import deque
import torch
from torch.utils.data import Dataset

from czf.pb import czf_pb2


class ReplayBuffer(Dataset):
    '''Replay Buffer'''
    def __init__(self, observation_shape, capacity, train_freq):
        self._observation_shape = observation_shape
        self._train_freq = train_freq
        self._buffer = deque(maxlen=capacity)
        self._num_games = 0
        self._ready = False

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]

    def is_ready(self):
        '''check if replay buffer is ready'''
        if self._ready:
            self._ready = False
            return True
        return False

    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''add trajectory to the replay buffer'''
        print('add', len(trajectory.states), 'positions')
        for state in trajectory.states:
            self._num_games += 1
            return
            # TODO
            observation_tensor = torch.tensor(state.observation_tensor).view(
                self._observation_shape)
            policy = torch.tensor(state.evaluation.policy)
            #value = torch.tensor(trajectory.returns)

            self._buffer.append((observation_tensor, policy, value))

        if self._num_games >= self._train_freq:
            self._ready = True
            self._num_games -= self._train_freq