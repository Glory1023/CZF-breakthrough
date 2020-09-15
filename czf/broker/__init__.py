'''CZF Broker'''
import argparse
import asyncio
import zmq.asyncio

from czf.broker.broker import Broker


async def main():
    '''czf.broker main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-l',
                        '--listen',
                        type=int,
                        required=True,
                        metavar='port',
                        help='broker listen port')
    args = parser.parse_args()
    broker = Broker(args)
    await broker.loop()


def run_main():
    '''Run main program in asyncio'''
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
