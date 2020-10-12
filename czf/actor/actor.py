'''CZF Actor'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import platform
import tempfile
import numpy as np
import czf_env

from czf.utils import get_zmq_dealer, RemoteModelManager
from czf.pb import czf_pb2


class Actor:
    '''Actor'''
    def __init__(self, args, worker_manager):
        #self.game = czf_env.load_game(args.game)
        self.num_actons = 9  # TODO
        # job queue
        self.worker_manager = worker_manager
        self.jobs = asyncio.Queue()
        # connect to model provider
        self.model_manager = RemoteModelManager(
            identity=f'model-manager-{args.suffix}',
            upstream=args.upstream,
            cache_size=8,
        )
        # connect to broker
        self.node = czf_pb2.Node(
            hostname=platform.node(),
            identity=f'actor-{args.suffix}',
        )
        self.socket = get_zmq_dealer(
            identity=self.node.identity,
            remote_address=args.broker,
        )
        asyncio.create_task(self.__initialize())

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_job_response_loop(),
                             self._enqueue_loop(), self._dequeue_loop())

    async def _recv_job_response_loop(self):
        '''a loop to receive `JobResponse`'''
        while True:
            raw = await self.socket.recv()
            packet = czf_pb2.Packet.FromString(raw)
            print(packet)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job_response':
                asyncio.create_task(self.__on_job_response(
                    packet.job_response))

    async def _enqueue_loop(self):
        '''a loop to enqueue `Job` into WorkerManager'''
        while True:
            job = await self.jobs.get()
            job.workers[job.step].CopyFrom(self.node)
            job.step += 1
            state = job.payload.state
            # TODO: job.model
            self.worker_manager.enqueue_job(
                job,
                np.array(state.observation_tensor, dtype=np.float32),
                np.array(state.legal_actions, dtype=np.int32),
            )

    async def _dequeue_loop(self):
        '''a loop to dequeue from WorkerManager and send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while True:
            job, total_visits, visits = await loop.run_in_executor(
                executor, self.worker_manager.wait_dequeue_result)
            assert (total_visits == sum(visits.values()))
            policy = [0.] * self.num_actons
            for action, visit in visits.items():
                policy[action] = visit
            job.payload.state.evaluation.policy[:] = policy
            await self.__send_packet(czf_pb2.Packet(job=job))

    def __load_model(self, model: czf_pb2.Model):
        assert (len(model.blobs) == 1)
        blob = model.blobs[0]
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(blob)
            self.worker_manager.load_model(tmp_file.name)

    async def __initialize(self):
        '''initialize model and start to send job'''
        # load model
        model_version = await self.model_manager.get_latest_version('default')
        print('Request Model:', model_version)
        model = await self.model_manager.get(
            czf_pb2.Model(name='default', version=model_version))
        self.__load_model(model)
        print('Finish Load Model')
        # start to send job
        asyncio.create_task(self.__send_job_request())

    async def __on_job_response(self, job_response: czf_pb2.JobResponse):
        '''enqueue jobs into Queue and send `JobRequest`'''
        for job in job_response.jobs:
            await self.jobs.put(job)
        await self.__send_job_request()

    async def __send_packet(self, packet: czf_pb2.Packet):
        '''helper to send a `Packet`'''
        await self.socket.send(packet.SerializeToString())

    async def __send_job_request(self, capacity=1280):
        '''helper to send a `JobRequest`'''
        packet = czf_pb2.Packet(job_request=czf_pb2.JobRequest(
            operation=czf_pb2.Job.Operation.MUZERO_SEARCH, capacity=capacity))
        await self.__send_packet(packet)
