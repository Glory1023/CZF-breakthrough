'''CZF Optimizer'''
import argparse
import asyncio
from pathlib import Path
import yaml
import zmq.asyncio

from czf.optimizer.optimizer import Optimizer


async def main():
    '''czf.optimizer main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-f', '--config', required=True, help='config file')
    parser.add_argument('-s',
                        '--storage-dir',
                        default=str(Path('storage').resolve()),
                        help='path to store model and trajectory')
    parser.add_argument('-l',
                        '--listen',
                        type=int,
                        required=True,
                        metavar='port',
                        help='optimizer listen port')
    parser.add_argument('-r',
                        '--restore',
                        action='store_true',
                        help='restore the latest checkpoint')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    optimizer = Optimizer(args, config)
    await optimizer.loop()


def run_main():
    '''Run main program in asyncio'''
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
