'''CZF Actor'''
import argparse
import asyncio
from uuid import uuid4
import czf_env
import torch
import zmq

from czf.actor.actor import Actor
from czf.actor import worker


async def main(args, worker_manager):
    '''czf.actor main program'''
    actor = Actor(args, worker_manager)
    await actor.loop()


def run_main():
    '''Run main program in asyncio'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-g',
                        '--game',
                        metavar='game_name',
                        help='czf_env game name')
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    parser.add_argument('-u',
                        '--upstream',
                        required=True,
                        metavar='host:port',
                        help='model provider address. e.g., 127.0.0.1:5566')
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        help='unique id of the actor',
                        default=uuid4().hex)
    parser.add_argument('-cpu', '--num_cpu_worker', type=int, default=1)
    parser.add_argument('-gpu', '--num_gpu_worker', type=int, default=1)
    args = parser.parse_args()

    worker_manager = worker.WorkerManager()
    # config
    game = czf_env.load_game(args.game)
    worker_manager.game_info.observation_shape = game.observation_tensor_shape
    worker_manager.game_info.state_shape = game.observation_tensor_shape
    worker_manager.game_info.num_actions = game.num_distinct_actions
    worker_manager.game_info.all_actions = list(
        range(game.num_distinct_actions))
    # run
    num_gpu = torch.cuda.device_count()
    worker_manager.run(args.num_cpu_worker, args.num_gpu_worker, num_gpu)
    try:
        asyncio.run(main(args, worker_manager))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
