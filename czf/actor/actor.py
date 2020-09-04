#! /usr/bin/env python3
import time
import asyncio
import zmq
import zmq.asyncio
import argparse
import platform
from uuid import uuid4

from czf.pb import czf_pb2


class Actor:
    def __init__(self, args):
        self.node = czf_pb2.Node(
            hostname=platform.node(),
            identity=f'actor-{args.suffix}'
        )
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.IDENTITY, self.node.identity)
        socket.connect(f'tcp://{args.broker}')
        self.socket = socket

        self.capacity = 1280

        asyncio.create_task(self.send_job_request())

    async def loop(self):
        await asyncio.gather(
            self.recv_loop()
        )

    async def recv_loop(self):
        while True:
            msg = await self.socket.recv()
            print(msg)

    async def send_packet(self, packet):
        raw = packet.SerializeToString()
        await self.socket.send(raw)

    async def send_job_request(self):
        packet = czf_pb2.Packet(
            job_request=czf_pb2.JobRequest(
                operation=czf_pb2.Job.Operation.ALPHAZERO_SEARCH,
                capacity=self.capacity
            )
        )
        await self.send_packet(packet)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', type=str, required=True)
    parser.add_argument('-b', '--broker', type=str, required=True)
    # parser.add_argument('-m', '--model-provider', type=str, required=True)
    parser.add_argument('--suffix', type=str, default=uuid4().hex)
    args = parser.parse_args()

    actor = Actor(args)
    await actor.loop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')
