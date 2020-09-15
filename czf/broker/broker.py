'''CZF Broker'''
import asyncio
from collections import defaultdict
import time
import zmq
import zmq.asyncio

from czf.pb import czf_pb2


class Broker:
    def __init__(self, args):
        # self.jobs[operation]
        self.jobs = defaultdict(asyncio.Queue)

        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f'tcp://*:{args.listen}')
        self.socket = socket

    async def loop(self):
        await asyncio.gather(
            self.recv_loop()
        )

    async def recv_loop(self):
        while True:
            identity, raw = await self.socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            print(packet)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job':
                job = packet.job
                if job.step == len(job.procedure):
                    self.socket.send_multipart([job.initiator.identity.encode(), raw])
                else:
                    operation = job.procedure[job.step]
                    await self.jobs[operation].put(job)
            elif packet_type == 'job_request':
                job_request = packet.job_request
                asyncio.create_task(self.dispatch_jobs(identity, job_request))

    async def dispatch_jobs(self, worker, job_request, wait_time=0.2):
        job_queue = self.jobs[job_request.operation]
        packet = czf_pb2.Packet(
            job_response=czf_pb2.JobResponse(
                jobs=[await job_queue.get()]
            )
        )
        deadline = time.time() + wait_time
        while (timeout := deadline - time.time()) > 0:
            try:
                job = await asyncio.wait_for(job_queue.get(), timeout=timeout)
                packet.job_response.jobs.append(job)
            except asyncio.exceptions.TimeoutError:
                break
        await self.socket.send_multipart([worker, packet.SerializeToString()])
