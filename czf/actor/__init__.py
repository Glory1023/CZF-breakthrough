'''CZF Actor'''
import argparse
import asyncio
import os
from pathlib import Path
import random
from uuid import uuid4

import torch
import yaml
import zmq

from czf.actor.actor import Actor


def create_worker_manager(args, config):
    assert config['algorithm'] in ('AlphaZero', 'MuZero')
    '''create a worker manager from args and config'''
    if config['algorithm'] == 'AlphaZero':
        # import worker here since importing both two workers will encounter protobuf error
        from czf.actor import alphazero_worker
        game_config = config['game']
        mcts_config = config['mcts']
        manager = alphazero_worker.WorkerManager()
        manager.load_game(game_config['name'])
        manager.worker_option.seed = args.seed
        manager.worker_option.timeout_us = args.gpu_timeout
        manager.worker_option.batch_size = args.batch_size
        manager.worker_option.num_sampled_transformations = mcts_config[
            'num_sampled_transformations']
    elif config['algorithm'] == 'MuZero':
        # import worker here since importing both two workers will encounter protobuf error
        use_gumbel = ('gumbel' in config['mcts'])
        if use_gumbel:
            from czf.actor import gumbel_muzero_worker
            manager = gumbel_muzero_worker.WorkerManager()
        else:
            from czf.actor import muzero_worker
            manager = muzero_worker.WorkerManager()
        # GameInfo
        model_config = config['model']
        game_config = config['game']
        obs_config = game_config['observation']
        frame_stack = obs_config['frame_stack']
        channel = obs_config['channel']
        spatial_shape = obs_config['spatial_shape']
        if frame_stack > 0:
            manager.game_info.observation_shape = [frame_stack * (channel + 1), *spatial_shape]
        else:
            manager.game_info.observation_shape = [channel, *spatial_shape]
        manager.game_info.state_shape = [
            config['model']['h_channels'], *game_config['state_spatial_shape']
        ]
        manager.game_info.num_actions = game_config['actions']
        manager.game_info.all_actions = list(range(game_config['actions']))
        manager.game_info.num_chance_outcomes = model_config.get('codebook_size', 0)
        manager.game_info.all_chance_outcomes = list(range(model_config.get('codebook_size', 0)))
        manager.game_info.is_two_player = (game_config.get('num_player', 2) == 2)
        manager.game_info.is_stochastic = game_config['is_stochastic']
        # JobOption
        manager.worker_option.seed = args.seed
        manager.worker_option.timeout_us = args.gpu_timeout
        manager.worker_option.batch_size = args.batch_size
    return manager


async def main(args):
    '''czf.actor main program'''
    config = yaml.safe_load(Path(args.config).read_text())
    worker_manager = create_worker_manager(args, config)
    actor = Actor(args, config, worker_manager)
    try:
        await actor.loop()
    except asyncio.CancelledError:
        worker_manager.terminate()


def run_main():
    '''Run main program in asyncio'''
    num_cpu = os.cpu_count()
    num_gpu = torch.cuda.device_count()
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-f', '--config', required=True, help='config file')
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    parser.add_argument('-u',
                        '--upstream',
                        required=True,
                        metavar='host:port',
                        help='model provider address. e.g., 127.0.0.1:5577')
    parser.add_argument('--eval', nargs='?', const='1P', help='evaluation mode (default: 1P)')
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        default=uuid4().hex,
                        help='unique id of the actor')
    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        default=2048,
                        help='max batch size per gpu worker')
    parser.add_argument('--seed',
                        type=int,
                        default=random.randint(0, 2**64),
                        help='random number seed of the actor')
    parser.add_argument('--gpu-timeout',
                        type=int,
                        default=1000,
                        help='GPU wait max timeout (default: %(default)s microseconds)')
    parser.add_argument('--num_manager',
                        type=int,
                        default=1,
                        help='Number of manager (default: %(default)s)')
    parser.add_argument('-cpu',
                        '--num_cpu_worker',
                        type=int,
                        default=num_cpu,
                        help='Number of cpu worker per manager (default: %(default)s)')
    parser.add_argument('-gpu',
                        '--num_gpu_worker',
                        type=int,
                        default=num_gpu,
                        help='Number of gpu worker per manager (default: %(default)s)')
    parser.add_argument('-gpu-root',
                        '--num_gpu_root_worker',
                        type=int,
                        default=num_gpu,
                        help='Number of gpu root worker per manager (default: %(default)s)')
    parser.add_argument('--num_gpu',
                        type=int,
                        default=num_gpu,
                        help='Number of gpu (default: %(default)s)')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
