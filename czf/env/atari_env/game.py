'''CZF AtariGame & AtariState'''
from collections import deque

import numpy as np
import gym
from gym.wrappers import AtariPreprocessing


class AtariState:
    '''Atari state wrapper'''
    def __init__(self, name, frame_stack, noop_max, video_dir):
        env = gym.make(name)
        env = AtariPreprocessing(
            env,
            noop_max=noop_max,
            screen_size=96,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=True,
        )
        if video_dir is not None:
            env = gym.wrappers.Monitor(
                env,
                video_dir,
                video_callable=lambda _: True,
                force=True,
            )
        self._env = env
        self._buffer = deque(maxlen=frame_stack) if frame_stack > 0 else None
        self._num_action = env.action_space.n
        self._all_actions = list(range(self._num_action))
        self._feat, self._done, self._reward = None, None, None
        self.reset()

    def _stack_action_observation(self, action, obs):
        '''Stack action and observation (4 x H x W)'''
        # HWC => CHW
        action_plane = np.full((1, *obs.shape[1:]), (action + 1) / self._num_action,
                               dtype=np.float32)
        # 4 (action, RGB) x H x W
        return np.vstack((action_plane, obs))

    def apply_action(self, action):
        '''Apply action to current state'''
        obs, reward, self._done, _ = self._env.step(action)
        obs = np.array(obs, dtype=np.float32).transpose((2, 0, 1))
        self._reward = [reward]
        if self._buffer is not None:
            self._feat = self._stack_action_observation(action, obs)
            self._buffer.append(self._feat)
        else:
            self._feat = obs

    def reset(self):
        '''reset the environment'''
        obs = self._env.reset()
        obs = np.array(obs, dtype=np.float32).transpose((2, 0, 1))
        if self._buffer is not None:
            self._feat = self._stack_action_observation(-1, obs)
            for _ in range(self._buffer.maxlen):
                self._buffer.append(self._feat)
        else:
            self._feat = obs
        self._reward, self._done = [0.], False

    @property
    def rewards(self):
        '''Returns reward from the most recent state transition
        (s, a, s') for all players'''
        return self._reward

    @property
    def is_terminal(self):
        '''Returns true if current state is terminal'''
        return self._done

    @property
    def current_player(self):
        '''Returns the player to move'''
        return 0

    @property
    def legal_actions(self):
        '''Return all legal actions for current player'''
        return self._all_actions

    @property
    def observation_tensor(self):
        '''Returns the input tensor for the neural network'''
        if self._buffer is not None:
            return np.vstack(self._buffer).flatten()
        return self._feat.flatten()

    @property
    def feature_tensor(self):
        '''Returns the feature tensor for the replay buffer'''
        return self._feat.flatten()

    @property
    def is_chance_node(self):
        '''Returns if current state is chance node'''
        return False

    @property
    def legal_chance_outcome_probs(self):
        '''Return all legal chance outcomes with probabilities'''
        return [(0, 1.0)]

    @property
    def current_stage(self):
        '''Returns the stage'''
        return 0


class AtariGame:
    '''Atari game wrapper'''
    def __init__(self, name, frame_stack, noop_max):
        self._name = name
        self._frame_stack = frame_stack
        self._noop_max = noop_max

    def new_initial_state(self, video_dir=None):
        '''Returns a newly allocated initial state'''
        return AtariState(self._name, self._frame_stack, self._noop_max, video_dir)

    @property
    def name(self):
        '''Returns the Atari gym id'''
        return self._name

    @property
    def num_players(self):
        '''The number of players in this game'''
        return 1

    @property
    def num_distinct_actions(self):
        '''Maximum number of distinct actions in the game for any one player'''
        raise NotImplementedError
        #return self._env.action_space.n

    @property
    def observation_tensor_shape(self):
        '''Describes the structure of the observation representation in
        a tuple (frame_stack, channel, (height, width)). The returned value
        will be used to construct the neural network
        '''
        raise NotImplementedError
        #return self._env.observation_space.shape

    @property
    def num_chance_outcomes(self):
        '''Maximum number of chance outcomes'''
        return 0
