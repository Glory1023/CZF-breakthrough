'''CZF Learner'''
import argparse
import asyncio
from pathlib import Path
import shutil
import yaml
import zmq.asyncio

from czf.learner.learner import Learner


async def main():
    '''czf.learner main program'''
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
                        help='learner listen port')
    parser.add_argument('-r',
                        '--restore',
                        action='store_true',
                        help='restore the latest checkpoint')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.config), Path(args.storage_dir) / 'config.yaml')
    learner = Learner(args, config)
    await learner.loop()


def run_main():
    '''Run main program in asyncio'''
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
