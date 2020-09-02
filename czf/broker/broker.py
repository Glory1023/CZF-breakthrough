#! /usr/bin/env python3
import time
import zmq
import asyncio
import zmq.asyncio

import czf_pb2


class Broker:
    def __init__(self):
        self.preprocessors = dict()
        self.self_players = dict()

        self.preprocess_requests = asyncio.Queue()
        self.search_requests = asyncio.Queue()

        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind('tcp://*:5555')
        self.socket = socket

    async def loop(self):
        await asyncio.gather(
            self.recv_loop(),
            self.request_dispatcher(self.preprocess_requests, self.preprocessors),
            self.request_dispatcher(self.search_requests, self.self_players)
        )

    async def recv_loop(self):
        while True:
            identity, raw = await self.socket.recv_multipart()
            packet = czf_pb2.Packet()
            packet.ParseFromString(raw)

            packet_type = packet.WhichOneof('payload')

            if packet_type == 'heartbeat':
                pass
            elif packet_type == 'job_request':
                self.add_worker(identity, packet.job_request)
            elif packet_type == 'preprocess_request':
                self.preprocess_requests.put_nowait(raw)
            elif packet_type == 'search_request':
                self.search_requests.put_nowait(raw)
                print(packet)

    async def request_dispatcher(self, request_queue, worker_dict, wait_time=0.2):
        while True:
            requests = [await request_queue.get()]
            deadline = time.time() + wait_time

            while (timeout := deadline - time.time()) > 0:
                try:
                    request = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                    requests.append(request)
                except asyncio.exceptions.TimeoutError:
                    break

            workers = sorted(worker_dict.items(), key=lambda item: (-item[1].efficiency, item[1].vacancy))
            for identity, status in workers:
                num_requests = min(status.vacancy, len(requests))
                if num_requests <= 0:
                    break
                for _ in range(num_requests):
                    request = requests.pop(0)
                    await self.socket.send_multipart([identity, request])
                worker_dict.pop(identity)

    def add_worker(self, identity, job_request):
        if job_request.type == czf_pb2.JobRequest.Type.PREPROCESS:
            self.preprocessors[identity] = job_request.worker_status
        elif job_request.type == czf_pb2.JobRequest.Type.SELF_PLAY:
            self.self_players[identity] = job_request.worker_status


async def main():
    broker = Broker()
    await broker.loop()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')
