'''CZF atari_env'''
import gym
from czf.game_server.atari_env.game import AtariGame


def load_game(name: str):
    '''Load AtariGame by name'''
    return AtariGame(name)


def available_games():
    '''List all available Atari games'''
    return [
        x.id for x in gym.envs.registry.all()
        if 'NoFrameskip-v4' in x.id and '-ram' not in x.id
    ]
