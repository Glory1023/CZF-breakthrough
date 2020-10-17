'''CZF Actor'''
import argparse
import asyncio
import os
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
                        help='unique id of the actor',
                        default=uuid4().hex)
    parser.add_argument('-cpu', '--num_cpu_worker', type=int, default=num_cpu)
    parser.add_argument('-gpu', '--num_gpu_worker', type=int, default=num_gpu)
    parser.add_argument('-gpu-root',
                        '--num_gpu_root_worker',
                        type=int,
                        default=1)
    args = parser.parse_args()
    # WorkerManager
    worker_manager = worker.WorkerManager()
    config = yaml.safe_load(Path(args.config).read_text())
    # GameInfo
    worker_manager.game_info.observation_shape = config['game'][
        'observation_shape']
    worker_manager.game_info.state_shape = config['game']['state_shape']
    worker_manager.game_info.num_actions = config['game']['actions']
    worker_manager.game_info.all_actions = list(
        range(config['game']['actions']))
    worker_manager.game_info.two_player = (config['game'].get('num_player',
                                                              2) == 2)
    # JobOption
    worker_manager.job_option.seed = config['mcts'].get('seed', 1)
    worker_manager.job_option.timeout_us = config['mcts'].get(
        'timeout_us', 1000)
    worker_manager.job_option.batch_size = config['mcts']['batch_size']
    worker_manager.job_option.simulation_count = config['mcts'][
        'simulation_count']
    # MctsOption
    worker_manager.mcts_option.C_PUCT = config['mcts']['c_puct']
    worker_manager.mcts_option.dirichlet_alpha = config['mcts']['dirichlet'][
        'alpha']
    worker_manager.mcts_option.dirichlet_epsilon = config['mcts']['dirichlet'][
        'epsilon']
    worker_manager.mcts_option.discount = config['mcts'].get('discount', 1.)

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
