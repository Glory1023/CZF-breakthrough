#! /usr/bin/env python3
import time
import asyncio
import zmq
import zmq.asyncio
import argparse
from uuid import uuid4

from czf.pb import czf_pb2


class Preprocessor:
    def __init__(self, suffix):
        self.identity = f'preprocessor-{suffix}'
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.IDENTITY, self.identity)
        socket.connect('tcp://localhost:5555')
        self.socket = socket

        self.capacity = 128
        self.efficiency = 1.5
        self.model_version = -1

        self.preprocess_requests = asyncio.Queue()
        asyncio.create_task(self.send_job_request())

    async def loop(self):
        await asyncio.gather(
            self.recv_loop(),
            self.job_scheduler()
        )

    async def heartbeat_loop(self):
        packet = czf_pb2.Packet()
        packet.heartbeat.SetInParent()
        raw = packet.SerializeToString()
        while True:
            await self.socket.send(raw)
            await asyncio.sleep(1)

    async def recv_loop(self):
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet()
            packet.ParseFromString(raw)

            if packet.WhichOneof('payload') == 'preprocess_request':
                self.preprocess_requests.put_nowait(packet.preprocess_request)

    async def send_job_request(self):
        packet = czf_pb2.Packet(
            job_request=czf_pb2.JobRequest(
                operation=czf_pb2.Job.Operation.MUZERO_PREPROCESS
            )
        )
        await self.send_packet(packet)

    async def job_scheduler(self):
        while True:
            preprocess_requests = [await self.preprocess_requests.get()]
            deadline = time.time() + 0.2

            while (timeout := deadline - time.time()) > 0:
                try:
                    preprocess_request = await asyncio.wait_for(self.preprocess_requests.get(), timeout=timeout)
                    preprocess_requests.append(preprocess_request)
                except asyncio.exceptions.TimeoutError:
                    break

            observations = [preprocess_request.observation for preprocess_request in preprocess_requests]
            states = self.preprocess(observations)

            for preprocess_request, state in zip(preprocess_requests, states):
                packet = czf_pb2.Packet()
                search_request = packet.search_request
                search_request.game_origin.CopyFrom(preprocess_request.game_origin)
                search_request.state = state
                search_request.legal_actions[:] = preprocess_request.legal_actions
                await self.send_packet(packet)

            await self.send_job_request()

    async def send_packet(self, packet):
        raw = packet.SerializeToString()
        await self.socket.send(raw)

    def preprocess(self, observations):
        return [b'preprocess(' + observation + b')' for observation in observations]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default=uuid4().hex)
    args = parser.parse_args()

    preprocessor = Preprocessor(args.suffix)
    await preprocessor.loop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')
