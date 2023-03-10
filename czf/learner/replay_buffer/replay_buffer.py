'''CZF Replay Buffer'''
from dataclasses import dataclass
import pickle
from torch.utils.data import Dataset
import zstandard as zstd

from czf.pb import czf_pb2


@dataclass
class Statistics:
    '''Statistics for a :class:`ReplayBuffer`'''
    num_games: int  # get_target_dist
    num_states: int  # total number of states
    game_steps: list  # a list of total number of game steps
    player_returns: list  # returns for each player


class ReplayBuffer(Dataset):
    '''ReplayBuffer is used to store and sample transitions.'''
    def __init__(
        self,
        num_player,
        states_to_train,
        sequences_to_train,
        sample_ratio,
        sample_states,
        capacity,
    ):
        self._num_player = num_player
        assert states_to_train != sequences_to_train, 'the two options are disjoint.'
        self._states_to_train = states_to_train
        self._sequences_to_train = sequences_to_train
        assert sample_ratio != sample_states, 'the two options are disjoint.'
        self._sample_ratio = sample_ratio
        self._sample_states = sample_states
        self._capacity = capacity
        self._num_states = 0
        self._num_games = 0
        self._ready = False
        self._trajectory_to_save = []
        self._cctx_trajectory = zstd.ZstdCompressor()
        self._dctx_trajectory = zstd.ZstdDecompressor()
        self.reset_statistics()

    def __len__(self):
        raise NotImplementedError('')

    def __getitem__(self, index):
        raise NotImplementedError('')

    def _extend(self, priorities, trajectories):
        raise NotImplementedError('')

    def get_weights(self):
        raise NotImplementedError('')

    def add_trajectory(self, trajectory: tuple):
        '''Add a trajectory and its statistics'''
        stats, priorities, trajectories = trajectory
        self._num_states += stats.num_states
        self._num_games += stats.num_games
        # update statistics
        self._statistics.num_games += stats.num_games
        self._statistics.num_states += stats.num_states
        self._statistics.game_steps.extend(stats.game_steps)
        for player, player_returns in enumerate(stats.player_returns):
            self._statistics.player_returns[player].extend(player_returns)

        # add trajectory to buffer
        self._extend(priorities, trajectories)
        self._trajectory_to_save.append(trajectory)

        # train the model when there are N newly generated states
        if self._states_to_train is not None:
            if self._num_states >= self._states_to_train:
                self._ready = True
            return stats.num_states
        # train the model when there are N newly generated sequences
        else:  #if self._sequences_to_train is not None:
            if self._num_games >= self._sequences_to_train:
                self._ready = True
            return stats.num_games

    def get_num_to_add(self):
        '''Get number of states or sequences needed for next training iteration'''
        if self._states_to_train is not None:
            return self._states_to_train - self._num_states
        # if self._sequences_to_train is not None:
        return self._sequences_to_train - self._num_games

    def get_states_to_train(self):
        '''Get number of states or sequences needed for current training iteration'''
        if self._ready:
            self._ready = False
            if self._states_to_train is not None:
                num_states = self._states_to_train
                self._num_states -= self._states_to_train
                self._num_games = 0
            else:  #if self._sequences_to_train is not None:
                num_states = self._num_states
                self._num_states = 0
                self._num_games -= self._sequences_to_train
            if self._sample_ratio is not None:
                return int(num_states * self._sample_ratio)
            return self._sample_states
        return 0

    def get_statistics(self):
        '''Returns :class:`Statistics` of recent trajectories'''
        return self._statistics

    def reset_statistics(self):
        '''Reset the :class:`Statistics` information of recent trajectories'''
        self._statistics = Statistics(
            num_games=0,
            num_states=0,
            game_steps=[],
            player_returns=[[] for _ in range(self._num_player)],
        )

    def save_trajectory(self, path, iteration):
        '''Save all trajectories to the `path` with compression, and clear up all trajactories'''
        '''Save all trajectories to the `path` with compression, and clear up all trajactories'''
        print(len(self._trajectory_to_save))
        serialized = pickle.dumps(self._trajectory_to_save)
        compressed = self._cctx_trajectory.compress(serialized)
        trajectory_path = path / f'{iteration:05d}.zst'
        trajectory_path.write_bytes(compressed)
        self._trajectory_to_save = []

    def restore_trajectory(self, path, end_iteration):
        num_states = 0
        start_iteration = 0
        # calculate the range of iterations to restore the full replay buffer
        for iteration in range(end_iteration, -1, -1):
            trajectory_path = path / f'{iteration:05d}.zst'
            compressed = trajectory_path.read_bytes()
            decompressed = self._dctx_trajectory.decompress(compressed)
            trajectories = pickle.loads(decompressed)
            for trajectory in trajectories:
                stats, _, _ = trajectory
                num_states += stats.num_states
            print(iteration, num_states, self._capacity)
            if num_states >= self._capacity:
                start_iteration = iteration
                break

        print(f'Restore trajectory from {start_iteration} to {end_iteration} iteration')
        for iteration in range(start_iteration, end_iteration + 1):
            trajectory_path = path / f'{iteration:05d}.zst'
            compressed = trajectory_path.read_bytes()
            decompressed = self._dctx_trajectory.decompress(compressed)
            trajectories = pickle.loads(decompressed)
            for trajectory in trajectories:
                self.add_trajectory(trajectory)
            print(iteration, len(self))
        self.reset_statistics()
