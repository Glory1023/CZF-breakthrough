'''CZF Broker'''
import asyncio
from collections import defaultdict
import time

from czf.pb import czf_pb2
from czf.utils import get_zmq_router


class Broker:
    '''Broker'''
    def __init__(self, args):
        # self._jobs[operation]
        self._jobs = defaultdict(asyncio.Queue)
        self._socket = get_zmq_router(listen_port=args.listen)

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop())

    async def _recv_loop(self):
        '''a loop to receive `Job` and `JobRequest`'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job':
                job = packet.job
                if job.step == len(job.procedure):
                    await self.__send_raw(job.initiator.identity.encode(), raw)
                else:
                    operation = job.procedure[job.step]
                    await self._jobs[operation].put(job)
            elif packet_type == 'job_request':
                job_request = packet.job_request
                asyncio.create_task(self.__dispatch_jobs(
                    identity, job_request))

    async def __dispatch_jobs(self, worker, job_request, wait_time=0.2):
        '''helper to send a `JobBatch`'''
        job_queue = self._jobs[job_request.operation]
        packet = czf_pb2.Packet(job_batch=czf_pb2.JobBatch(
            jobs=[await job_queue.get()]))
        deadline = time.time() + wait_time
        while (timeout := deadline - time.time()) > 0:
            try:
                job = await asyncio.wait_for(job_queue.get(), timeout=timeout)
                packet.job_batch.jobs.append(job)
            except asyncio.exceptions.TimeoutError:
                break
        await self.__send_packet(worker, packet)

    async def __send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self.__send_raw(identity, packet.SerializeToString())

    async def __send_raw(self, identity: bytes, raw: bytes):
        '''helper to send a zmq message'''
        await self._socket.send_multipart([identity, raw])
