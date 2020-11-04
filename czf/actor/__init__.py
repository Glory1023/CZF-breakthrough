'''CZF Actor'''
import argparse
import asyncio
import os
import random
from uuid import uuid4
from pathlib import Path
import torch
import yaml
import zmq

from czf.actor.actor import Actor
from czf.actor import worker


async def main(args, worker_manager):
    '''czf.actor main program'''
    actor = Actor(args, worker_manager)
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
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        default=uuid4().hex,
                        help='unique id of the actor')
    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        default=2048,
                        help='GPU max batch size')
    parser.add_argument('--seed',
                        type=int,
                        default=random.randint(0, 2**64),
                        help='random number seed of the actor')
    parser.add_argument(
        '--gpu-timeout',
        type=int,
        default=1000,
        help='GPU wait max timeout (default: %(default)s microseconds)')
    parser.add_argument(
        '-cpu',
        '--num_cpu_worker',
        type=int,
        default=num_cpu,
        help='Total number of cpu worker (default: %(default)s)')
    parser.add_argument(
        '-gpu',
        '--num_gpu_worker',
        type=int,
        default=num_gpu,
        help='Total number of gpu worker (default: %(default)s)')
    parser.add_argument(
        '-gpu-root',
        '--num_gpu_root_worker',
        type=int,
        default=1,
        help='Total number of gpu root worker (default: %(default)s)')
    args = parser.parse_args()
    # WorkerManager
    worker_manager = worker.WorkerManager()
    config = yaml.safe_load(Path(args.config).read_text())
    # GameInfo
    game_config = config['game']
    worker_manager.game_info.observation_shape = game_config[
        'observation_shape']
    worker_manager.game_info.state_shape = game_config['state_shape']
    worker_manager.game_info.num_actions = game_config['actions']
    worker_manager.game_info.all_actions = list(range(game_config['actions']))
    worker_manager.game_info.two_player = (game_config.get('num_player',
                                                           2) == 2)
    # JobOption
    worker_manager.worker_option.seed = args.seed
    worker_manager.worker_option.timeout_us = args.gpu_timeout
    worker_manager.worker_option.batch_size = args.batch_size

    # run
    worker_manager.run(
        num_cpu_worker=args.num_cpu_worker,
        num_gpu_worker=args.num_gpu_worker,
        num_gpu_root_worker=args.num_gpu_root_worker,
        num_gpu=num_gpu,
    )
    try:
        asyncio.run(main(args, worker_manager))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
