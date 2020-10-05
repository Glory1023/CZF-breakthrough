'''CZF Actor'''
import argparse
import asyncio
from uuid import uuid4
import zmq

from czf.actor.actor import Actor
from czf.actor import worker


async def main(manager):
    '''czf.actor main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-g',
                        '--game',
                        required=True,
                        metavar='game_name',
                        help='game name')
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    # parser.add_argument('-m', '--model-provider', type=str, required=True)
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        help='unique id of the actor',
                        default=uuid4().hex)
    args = parser.parse_args()

    actor = Actor(args, manager)
    await actor.loop()


def run_main():
    '''Run main program in asyncio'''
    num_cpu_workers, num_gpu_workers = 1, 1
    manager = worker.WorkerManager()
    for _ in range(num_cpu_workers):
        manager.register_worker(worker.WorkerCPU())
    for _ in range(num_gpu_workers):
        manager.register_worker(worker.WorkerGPU())
    manager.run()
    try:
        asyncio.run(main(manager))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
