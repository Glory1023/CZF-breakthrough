#! /usr/bin/env python3
import zmq
import asyncio
import zmq.asyncio
import argparse
from uuid import uuid4

from .model_manager import LocalModelManager, RemoteModelManager
import czf_pb2


class ModelProvider:
    def __init__(self, args):
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f'tcp://*:{args.listen}')
        self.provider = socket
        self.model_manager = LocalModelManager(args) if args.storage else RemoteModelManager(args)

    async def loop(self):
        while True:
            identity, raw = await self.provider.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')

            if packet_type == 'model_request':
                asyncio.create_task(self.send_model(identity, packet.model_request))

    async def send_model(self, identity: bytes, model: czf_pb2.Model):
        packet = czf_pb2.Packet()
        model = await self.model_manager.get(model)
        packet.model_response.CopyFrom(model)
        raw = packet.SerializeToString()
        self.provider.send_multipart([identity, raw])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cache-size', type=int, default=8)
    parser.add_argument('-l', '--listen', type=int, required=True)
    parser.add_argument('--suffix', type=str, default=uuid4().hex)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--storage', type=str)
    group.add_argument('-u', '--upstream', type=str)

    args = parser.parse_args()

    model_provider = ModelProvider(args)
    await model_provider.loop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')
