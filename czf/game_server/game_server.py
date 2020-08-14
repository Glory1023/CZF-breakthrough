#! /usr/bin/env python3
import zmq
import asyncio
import zmq.asyncio
import argparse
from uuid import uuid4

import czf_pb2


class GameServer:
    def __init__(self, suffix):
        self.identity = f"game-server-{suffix}"
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IDENTITY, self.identity.encode())
        socket.connect("tcp://localhost:5555")
        self.socket = socket

    async def loop(self):
        await asyncio.gather(
            self.send_loop(),
            self.recv_loop()
        )

    async def send_loop(self):
        while True:
            await asyncio.sleep(0.1)
            for i in range(8):
                packet = czf_pb2.Packet()
                preprocess_request = packet.preprocess_request
                preprocess_request.observation = uuid4().hex.encode()
                preprocess_request.legal_actions[:] = [1, 4, 5, 6, 8, 10]
                game_origin = preprocess_request.game_origin
                game_origin.node = self.identity
                game_origin.index = i
                await self.socket.send(packet.SerializeToString())

    async def recv_loop(self):
        while True:
            msg = await self.socket.recv()
            print(msg)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default=uuid4().hex)
    args = parser.parse_args()

    game_server = GameServer(args.suffix)
    await game_server.loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print("\rterminated by ctrl-c")
