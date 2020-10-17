'''CZF Dataloader'''
from collections import deque, namedtuple
from itertools import islice, zip_longest
import torch
from torch.utils.data import Dataset

from czf.pb import czf_pb2

Transition = namedtuple('Transition', [
    'observation',
    'action',
    'policy',
    'value',
    'reward',
    'is_terminal',
])


class RolloutBatch:
    '''Rollout Batch'''
    def __init__(self, data):
        transposed_data = list(self.__zip_discard(*data))
        self.observation = torch.stack(transposed_data[0], 0)
        self.transition = [torch.stack(d, 0) for d in transposed_data[1:] if d]

    @staticmethod
    def __zip_discard(*iterables, sentinel=None):
        '''zip longest and discard `sentinel`'''
        return [[entry for entry in iterable if entry is not sentinel]
                for iterable in zip_longest(*iterables, fillvalue=sentinel)]


class ReplayBuffer(Dataset):
    '''Replay Buffer'''
    def __init__(self, num_player, observation_shape, kstep, nstep,
                 discount_factor, capacity, train_freq):
        self._num_player = num_player
        self._observation_shape = observation_shape
        self._kstep = kstep
        self._nstep = nstep
        self._discount_factor = discount_factor
        self._train_freq = train_freq
        self._buffer = deque(maxlen=capacity)
        self._num_games = 0
        self._ready = False

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        rollout = list(islice(self._buffer, index, index + self._kstep))
        # o, [(p, v, m, a, r)]
        result = [rollout[0].observation]
        for transition in rollout:
            result.extend([transition.policy, transition.value])
            if transition.is_terminal:
                result.extend([torch.tensor(0), None, None])
                break
            result.extend(
                [torch.tensor(1), transition.action, transition.reward])
        result[-3:] = [torch.tensor(0), None, None]
        return result

    def is_ready(self):
        '''check if replay buffer is ready'''
        if self._ready:
            self._ready = False
            return True
        return False

    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''add trajectory to the replay buffer'''
        #print('add', len(trajectory.states), 'positions')
        self._num_games += len(trajectory.states)
        # from the terminal state to the initial state
        nstep, gamma = self._nstep, self._discount_factor
        discounted_return = [[0] for _ in range(self._num_player)]
        values = [0]
        for i, state in enumerate(reversed(trajectory.states)):
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
            observation = torch.tensor(state.observation_tensor).view(
                self._observation_shape)
            action = torch.tensor(state.transition.action)
            policy = torch.tensor(state.evaluation.policy)
            value = torch.tensor([nstep_value])
            reward = torch.tensor(
                [state.transition.rewards[state.transition.current_player]])
            # (o_t, p_t, v_t, a_{t+1}, r_{t+1}, is_terminal)
            self._buffer.append(
                Transition(
                    observation=observation,
                    action=action,
                    policy=policy,
                    value=value,
                    reward=reward,
                    is_terminal=(i == 0),
                ))

        if self._num_games >= self._train_freq:
            self._ready = True
            self._num_games -= self._train_freq
