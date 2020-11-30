'''CZF Broker'''
import argparse
import asyncio
import zmq.asyncio

from czf.broker.broker import Broker
from czf.broker.named_broker import NamedBroker


async def main():
    '''czf.broker main program'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-l',
                        '--listen',
                        type=int,
                        required=True,
                        metavar='port',
                        help='broker listen port. e.g., 5566')
    parser.add_argument('-a', help='actor identity e.g., actor-NV01')
    parser.add_argument('-g', help='game server identity e.g., gs-NV01')
    args = parser.parse_args()
    BrokerCls = NamedBroker if args.a and args.g else Broker
    broker = BrokerCls(args)
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
