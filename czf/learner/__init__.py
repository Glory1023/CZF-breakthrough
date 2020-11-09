'''CZF Learner'''
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import shutil
import yaml
import zmq.asyncio

from czf.learner.learner import Learner


async def main():
    '''czf.learner main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-f', '--config', required=True, help='config file')
    parser.add_argument('-l',
                        '--listen',
                        type=int,
                        required=True,
                        metavar='PORT',
                        help='learner listen port')
    parser.add_argument('-m',
                        '--model_name',
                        default='default',
                        help='default name of the model')
    parser.add_argument('-s',
                        '--storage-dir',
                        help='path to store model, trajectory, and log')
    parser.add_argument('-r',
                        '--restore',
                        nargs='?',
                        const='',
                        metavar='CHECKPOINT',
                        help='restore the latest checkpoint')
    parser.add_argument('--restore-buffer',
                        nargs='?',
                        const='',
                        metavar='TRAJECTORY_DIR',
                        help='restore the replay buffer')
    parser.add_argument('--pretrain-trajectory',
                        metavar='TRAJECTORY_DIR',
                        help='pretrain the trajectory')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    # default storage dir: `storage_{game}_{date}`
    if not args.storage_dir:
        game = config['game']['name']
        path = 'storage_' + game + '_' + datetime.today().strftime(
            '%Y%m%d_%H%M')
        args.storage_dir = str(Path(path).resolve())
    storage_path = Path(args.storage_dir)
    print('Storage path:', storage_path)
    # default restore checkpoint: `{storage}/checkpoint/{model_name}/latest.pt.zst`
    if args.restore == '':
        args.restore = storage_path / 'checkpoint' / args.model_name / 'latest.pt.zst'
    # default restore buffer: `{storage}/trajectory`
    if args.restore_buffer == '':
        args.restore_buffer = storage_path / 'trajectory'
    # copy config.yaml if not exists
    if not storage_path.exists():
        storage_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.config), storage_path / 'config.yaml')

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
