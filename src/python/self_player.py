#! /usr/bin/env python3
import time
import asyncio
import zmq
import zmq.asyncio
import argparse
from uuid import uuid4

import constants
import czf_pb2


class SelfPlayer:
    def __init__(self, suffix):
        self.identity = f"{constants.SELF_PLAYER_IDENTITY_PREFIX}-{suffix}".encode()
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IDENTITY, self.identity)
        socket.connect("tcp://localhost:5555")
        self.socket = socket

        self.capacity = 1280
        self.efficiency = 1.5
        self.model_version = -1

        self.search_requests = asyncio.Queue()
        asyncio.ensure_future(self.send_job_request())

    async def loop(self):
        await asyncio.gather(
            self.heartbeat_loop(),
            self.send_loop(),
            self.recv_loop()
        )

    async def heartbeat_loop(self):
        packet = czf_pb2.Packet()
        packet.heartbeat.SetInParent()
        raw = packet.SerializeToString()
        while True:
            await self.socket.send(raw)
            await asyncio.sleep(1)

    async def send_loop(self):
        while True:
            await asyncio.sleep(1)
            packet = czf_pb2.Packet()
            packet.worker_status.CopyFrom(self.status)
            await self.socket.send(packet.SerializeToString())

    async def recv_loop(self):
        while True:
            msg = await self.socket.recv()
            print(msg)

    async def send_job_request(self):
        packet = czf_pb2.Packet()
        job_request = packet.job_request
        job_request.type = czf_pb2.JobRequest.Type.SEARCH
        job_request.worker_status.vacancy = self.capacity - self.preprocess_requests.qsize()
        job_request.worker_status.efficiency = self.efficiency
        job_request.worker_status.model_version = self.model_version
        await self.send_packet(packet)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default=uuid4().hex)
    args = parser.parse_args()

    self_player = SelfPlayer(args.suffix)
    await self_player.loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print("\rterminated by ctrl-c")
