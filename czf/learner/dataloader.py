'''CZF Dataloader'''
from collections import Counter, deque, namedtuple
from itertools import islice, zip_longest
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


class RolloutBatch:
    '''Rollout Batch'''
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
        self._pb_trajectory_batch = czf_pb2.TrajectoryBatch()
        self._compressor = zstd.ZstdCompressor()
        self._num_games = 0
        self._ready = False

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        if self._buffer[index].is_terminal:
            if index == 0:
                index += 1
            else:
                index -= 1
        rollout = list(islice(self._buffer, index, index + self._kstep))
        observation, kstep, transitions = rollout[0].observation, 0, []
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
        '''check if replay buffer is ready'''
        if self._ready:
            self._ready = False
            return True
        return False

    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''add trajectory to the replay buffer'''
        #print('add', len(trajectory.states), 'positions')
        self._pb_trajectory_batch.trajectories.add().CopyFrom(trajectory)
        self._num_games += len(trajectory.states)
        # from the terminal state to the initial state
        nstep, gamma = self._nstep, self._discount_factor
        discounted_return = [[0] for _ in range(self._num_player)]
        values = [0]
        buffer = []
        for i, state in enumerate(reversed(trajectory.states)):
            # tensor
            observation = torch.tensor(state.observation_tensor).view(
                self._observation_shape)
            if i == 0:
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
        '''get statistics of recent trajectories'''
        size = len(self._buffer)
        values = Counter([
            str(float(transition.value))
            for transition in islice(self._buffer, size -
                                     self._train_freq, size)
            if transition.is_terminal
        ])
        total = sum(values.values())
        # TODO: average length of trajectories
        return {k: v / total for k, v in values.items()}

    def save_trajectory(self, path, iteration):
        '''save trajectory to path'''
        trajectory = self._pb_trajectory_batch.SerializeToString()
        compressed = self._compressor.compress(trajectory)
        trajectory_path = path / f'{iteration:05d}.pb.zst'
        trajectory_path.write_bytes(compressed)
        self._pb_trajectory_batch.Clear()
