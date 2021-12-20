'''CZF Replay Buffer'''
from dataclasses import dataclass
from torch.utils.data import Dataset
import zstandard as zstd
from czf.pb import czf_pb2


@dataclass
class Statistics:
    '''Statistics for a :class:`ReplayBuffer`

    :param num_games: total number of games
    :param num_states: total number of states
    :param game_steps: a list of total number of game steps
    :param player_returns: returns for each player
    '''
    num_games: int
    num_states: int
    game_steps: list
    player_returns: list


class ReplayBuffer(Dataset):
    '''ReplayBuffer is used to store and sample transitions.'''
    def __init__(
        self,
        num_player,
        states_to_train,
        sequences_to_train,
        sample_ratio,
        sample_states,
    ):
        self._num_player = num_player
        assert states_to_train != sequences_to_train, 'the two options are disjoint.'
        self._states_to_train = states_to_train
        self._sequences_to_train = sequences_to_train
        assert sample_ratio != sample_states, 'the two options are disjoint.'
        self._sample_ratio = sample_ratio
        self._sample_states = sample_states
        self._pb_trajectory_batch = czf_pb2.TrajectoryBatch()
        self._cctx_trajectory = zstd.ZstdCompressor()
        self._num_states = 0
        self._num_games = 0
        self._ready = False
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
        '''Save all trajectories to the `path` with compression,
        and clear up all trajactories'''
        trajectory = self._pb_trajectory_batch.SerializeToString()
        compressed = self._cctx_trajectory.compress(trajectory)
        trajectory_path = path / f'{iteration:05d}.pb.zst'
        trajectory_path.write_bytes(compressed)
        self._pb_trajectory_batch.Clear()
