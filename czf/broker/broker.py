'''CZF Broker'''
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random

from czf.pb import czf_pb2
from czf.utils import get_zmq_router


class Broker:
    '''Broker'''
    def __init__(self, args):
        self._peers = defaultdict(set)
        self._socket = get_zmq_router(listen_port=args.listen)
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._print_job_batch = args.print_job_batch

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop())

    async def _recv_loop(self):
        '''a loop to receive `Job`, `JobBatch`, and `JobRequest`'''
        print("Wait to recieve...")
        while True:
            identity, *raw = await self._socket.recv_multipart()
            if len(raw) != 1:  # prevent wrong connections (not from czf)
                continue
            raw = raw[0]
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job':
                print("Packet_type: job")
                asyncio.create_task(self._on_recv_job(packet.job))
            elif packet_type == 'job_batch':
                print("Packet_type: job_batch")
                for job in packet.job_batch.jobs:
                    asyncio.create_task(self._on_recv_job(job))
            elif packet_type == 'job_request':
                print("Packet_type: job_request. Identity:  ", identity)
                job_request = packet.job_request
                self._peers[job_request.operation].add(identity)
            elif packet_type == 'goodbye':
                if identity.decode('UTF-8')[:5] == 'actor':
                    # from actor
                    print("Actor leave. (identity: ", identity, ")")
                    self._peers[job_request.operation].remove(identity)
                else:
                    # from game server
                    print("Game-server leave. (identity: ", identity, ")")

    async def _on_recv_job(self, job: czf_pb2.Job):
        def get_worker(peers):
            while len(peers) == 0:
                pass
            return random.sample(peers, 1)[0]

        if job.step == len(job.procedure):
            if self._print_job_batch: print(">> game server")
            # send job back to game server
            packet = czf_pb2.Packet(job=job)
            await self._send_packet(job.initiator.identity.encode(), packet)
        else:
            # send job to one of matched actor
            operation = job.procedure[job.step]
            loop = asyncio.get_event_loop()
            if self._print_job_batch: print("<< actor operation: ", operation)
            worker = await loop.run_in_executor(self._executor, get_worker, self._peers[operation])
            packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(jobs=[job]))
            await self._send_packet(worker, packet)

    async def _send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        raw = packet.SerializeToString()
        await self._socket.send_multipart([identity, raw])
