'''CZF Actor'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import platform
import random
import czf_env
import zmq
import zmq.asyncio

from czf.pb import czf_pb2


class Actor:
    def __init__(self, args, manager):
        self.manager = manager
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = asyncio.Queue()
        self.game = czf_env.load_game(args.game)
        self.node = czf_pb2.Node(hostname=platform.node(),
                                 identity=f'actor-{args.suffix}')
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.IDENTITY, self.node.identity)
        socket.connect(f'tcp://{args.broker}')
        self.socket = socket

        self.capacity = 1280

        asyncio.create_task(self.send_job_request())

    async def loop(self):
        await asyncio.gather(self.recv_loop(), self.enqueue_loop(),
                             self.dequeue_loop())

    async def recv_loop(self):
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet.FromString(raw)
            print(packet)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job_response':
                asyncio.create_task(self.on_job_response(packet.job_response))

    async def enqueue_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            job = await self.jobs.get()
            job.workers[job.step].CopyFrom(self.node)
            job.step += 1
            state = job.payload.state
            self.manager.enqueue_job(job, np.array(state.observation_tensor),
                                     np.array(state.observation_tensor_shape),
                                     np.array(state.legal_actions))

    async def dequeue_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            job = await loop.run_in_executor(self.executor,
                                             self.manager.wait_dequeue_result)
            print(job)
            # job.payload.state.evaluation.policy = policy
            await self.send_packet(czf_pb2.Packet(job=job))

    async def on_job_response(self, job_response: czf_pb2.JobResponse) -> None:
        for job in job_response.jobs:
            await self.jobs.put(job)
        await self.send_job_request()

    async def send_packet(self, packet):
        raw = packet.SerializeToString()
        await self.socket.send(raw)

    async def send_job_request(self):
        packet = czf_pb2.Packet(job_request=czf_pb2.JobRequest(
            operation=czf_pb2.Job.Operation.ALPHAZERO_SEARCH,
            capacity=self.capacity))
        await self.send_packet(packet)
