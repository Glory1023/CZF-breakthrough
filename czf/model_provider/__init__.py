'''CZF ModelProvider'''
import argparse
import asyncio
from uuid import uuid4
import zmq.asyncio

from czf.model_provider.model_provider import ModelProvider


async def main(args):
    '''czf.model_provider main program'''
    model_provider = ModelProvider(args)
    await model_provider.loop()


def run_main():
    '''Run main program in asyncio'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-l',
                        '--listen',
                        type=int,
                        required=True,
                        metavar='PORT',
                        help='provider listen port')
    parser.add_argument('-c', '--cache-size', type=int, default=8, help='total cached models')
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        default=uuid4().hex,
                        help='unique id of the actor')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--storage', help='path to load model')
    group.add_argument('-u',
                       '--upstream',
                       metavar='host:port',
                       help='model provider address. e.g., 127.0.0.1:5577')
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
