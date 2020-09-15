'''CZF Game Server'''
import argparse
import asyncio
from uuid import uuid4
import zmq.asyncio

from czf.game_server.game_server import GameServer


async def main():
    '''czf.game_server main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    parser.add_argument('-g',
                        '--game',
                        required=True,
                        metavar='game_name',
                        help='game name')
    parser.add_argument('-n', '--num-env', type=int, required=True)
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        help='unique id of the game server',
                        default=uuid4().hex)
    args = parser.parse_args()

    game_server = GameServer(args)
    await game_server.loop()


def run_main():
    '''Run main program in asyncio'''
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
