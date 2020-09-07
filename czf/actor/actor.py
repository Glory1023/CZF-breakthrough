#! /usr/bin/env python3
import random
import asyncio
import zmq
import zmq.asyncio
import argparse
import platform
from uuid import uuid4


from czf.pb import czf_pb2
import czf_env


class Actor:
    def __init__(self, args):
        self.jobs = asyncio.Queue()

        self.game = czf_env.load_game(args.game)

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
            self.recv_loop(),
            self.search_loop()
        )

    async def recv_loop(self):
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet.FromString(raw)
            print(packet)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job_response':
                asyncio.create_task(self.on_job_response(packet.job_response))

    async def search_loop(self):
        while True:
            job = await self.jobs.get()
            job.workers[job.step].CopyFrom(self.node)
            job.step += 1
            policy = job.payload.state.evaluation.policy
            policy[:] = [random.random() for _ in range(self.game.num_distinct_actions)]
            policy_sum = sum(policy)
            for i in range(len(policy)):
                policy[i] /= policy_sum
            await self.send_packet(czf_pb2.Packet(job=job))

    async def on_job_response(self, job_response: czf_pb2.JobResponse) -> None:
        for job in job_response.jobs:
            await self.jobs.put(job)
        await self.send_job_request()

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
