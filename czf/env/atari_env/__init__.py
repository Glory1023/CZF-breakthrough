'''CZF atari_env'''
import gym
from czf.env.atari_env.game import AtariGame


def load_game(name: str, frame_stack=32, noop_max=0):
    '''Load AtariGame by name'''
    return AtariGame(name, frame_stack, noop_max)


def available_games():
    '''List all available Atari games'''
    return [
        x.id for x in gym.envs.registry.all()
        if 'NoFrameskip-v4' in x.id and '-ram' not in x.id
    ]
