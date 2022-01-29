'''CZF Broker'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import random

from czf.pb import czf_pb2
from czf.utils import get_zmq_router


class Broker:
    '''Broker'''
    def __init__(self, args):
        self._peers = defaultdict(set)
        self._socket = get_zmq_router(listen_port=args.listen)
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop())

    async def _recv_loop(self):
        '''a loop to receive `Job`, `JobBatch`, and `JobRequest`'''
        while True:
            identity, *raw = await self._socket.recv_multipart()
            if len(raw) != 1:  # prevent wrong connections (not from czf)
                continue
            raw = raw[0]
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job':
                asyncio.create_task(self.__on_recv_job(packet.job))
            elif packet_type == 'job_batch':
                for job in packet.job_batch.jobs:
                    asyncio.create_task(self.__on_recv_job(job))
            elif packet_type == 'job_request':
                job_request = packet.job_request
                self._peers[job_request.operation].add(identity)

    async def __on_recv_job(self, job: czf_pb2.Job):
        def get_worker(peers):
            while len(peers) == 0:
                pass
            return random.sample(peers, 1)[0]

        if job.step == len(job.procedure):
            packet = czf_pb2.Packet(job=job)
            await self.__send_packet(job.initiator.identity.encode(), packet)
        else:
            operation = job.procedure[job.step]
            loop = asyncio.get_event_loop()
            worker = await loop.run_in_executor(self._executor, get_worker, self._peers[operation])
            packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job]))
            await self.__send_packet(worker, packet)

    async def __send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        raw = packet.SerializeToString()
        await self._socket.send_multipart([identity, raw])
