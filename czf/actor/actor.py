'''CZF Actor'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import platform
import tempfile
import numpy as np

from czf.actor.worker import TreeOption
from czf.utils import get_zmq_dealer, RemoteModelManager
from czf.pb import czf_pb2


class Actor:
    '''Actor'''
    def __init__(self, args, worker_manager):
        self._node = czf_pb2.Node(
            hostname=platform.node(),
            identity=f'actor-{args.suffix}',
        )
        # Actor mode
        operation = {
            None: czf_pb2.Job.Operation.MUZERO_SEARCH,
            '1P': czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
            '2P': czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
        }
        if args.eval:
            assert args.eval in ('1P', '2P')
            print(f'[Evaluation {args.eval} Mode]', self._node.identity)
        else:
            print('[Training Mode]', self._node.identity)
        self._operation = operation[args.eval]
        # worker manager
        self._worker_manager = worker_manager
        worker_manager.run(
            num_cpu_worker=args.num_cpu_worker,
            num_gpu_worker=args.num_gpu_worker,
            num_gpu_root_worker=args.num_gpu_root_worker,
            num_gpu=args.num_gpu,
        )
        # job queue
        self._jobs = asyncio.Queue()
        # model
        self._model_info = czf_pb2.ModelInfo(name='default', version=-1)
        self._has_new_model = asyncio.Event()
        self._has_load_model = asyncio.Event()
        # connect to the remote model provider
        self._model_manager = RemoteModelManager(
            identity=f'model-manager-{args.suffix}',
            upstream=args.upstream,
            cache_size=8,
        )
        # connect to the broker
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )
        asyncio.create_task(self.__initialize())

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_job_batch_loop(), self._enqueue_loop(),
                             self._dequeue_loop(), self._load_model_loop())

    async def _recv_job_batch_loop(self):
        '''a loop to receive `JobBatch`'''
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            #print(packet)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'job_batch':
                asyncio.create_task(self.__on_job_batch(packet.job_batch))

    async def _enqueue_loop(self):
        '''a loop to enqueue `Job` into WorkerManager'''
        while True:
            job = await self._jobs.get()
            job.workers[job.step].CopyFrom(self._node)
            job.step += 1
            if not job.HasField('payload'):  # special job: flush model
                if self._model_info != job.model:
                    self._model_info.CopyFrom(job.model)
                    self._has_load_model.clear()
                    self._has_new_model.set()
                    await self._has_load_model.wait()
                await self.__send_job(job)
                continue
            state = job.payload.state
            # assert job.model.name == self._model_info.name
            if job.model.version > self._model_info.version:
                self._model_info.version = job.model.version
                self._has_new_model.set()
            # copy job option
            option = state.tree_option
            tree_option = TreeOption()
            tree_option.simulation_count = option.simulation_count
            tree_option.tree_min_value = option.tree_min_value
            tree_option.tree_max_value = option.tree_max_value
            tree_option.c_puct = option.c_puct
            tree_option.dirichlet_alpha = option.dirichlet_alpha
            tree_option.dirichlet_epsilon = option.dirichlet_epsilon
            tree_option.discount = option.discount
            # enqueue
            self._worker_manager.enqueue_job(
                job,
                np.array(state.observation_tensor, dtype=np.float32),
                np.array(state.legal_actions, dtype=np.int32),
                tree_option,
            )

    async def _dequeue_loop(self):
        '''a loop to dequeue from WorkerManager and send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while True:
            job, *result = await loop.run_in_executor(
                executor, self._worker_manager.wait_dequeue_result)
            if job is None:
                return
            # assert total_visits == sum(visits.values())
            value, total_visits, visits = result
            policy = [0.] * self._worker_manager.game_info.num_actions
            for action, visit in visits.items():
                policy[action] = visit / total_visits
            job.payload.state.evaluation.value = value
            job.payload.state.evaluation.policy[:] = policy
            asyncio.create_task(self.__send_job(job))

    async def _load_model_loop(self):
        '''a loop to load model'''
        while await self._has_new_model.wait():
            self._has_new_model.clear()
            model = await self._model_manager.get(self._model_info)
            self.__load_model(model)
            self._has_load_model.set()

    def __load_model(self, model: czf_pb2.Model):
        '''WorkerManager load model'''
        assert len(model.blobs) == 1
        print('load model', model.info.name, 'iteration', model.info.version)
        blob = model.blobs[0]
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(blob)
            self._worker_manager.load_from_file(tmp_file.name)

    async def __initialize(self):
        '''initialize model and start to send job'''
        # load the latest model
        self._model_info.version = await self._model_manager.get_latest_version(
            self._model_info.name)
        model = await self._model_manager.get(self._model_info)
        self.__load_model(model)
        # start to send job
        await self.__send_job_request()

    async def __on_job_batch(self, job_batch: czf_pb2.JobBatch):
        '''enqueue jobs into Queue and send `JobRequest`'''
        for job in job_batch.jobs:
            await self._jobs.put(job)
        await self.__send_job_request()

    async def __send_job(self, job):
        '''helper to send a `Job`'''
        await self.__send_packet(czf_pb2.Packet(job=job))

    async def __send_job_request(self, capacity=1280):
        '''helper to send a `JobRequest`'''
        packet = czf_pb2.Packet(job_request=czf_pb2.JobRequest(
            operation=self._operation, capacity=capacity))
        await self.__send_packet(packet)

    async def __send_packet(self, packet: czf_pb2.Packet):
        '''helper to send a `Packet`'''
        await self._broker.send(packet.SerializeToString())
