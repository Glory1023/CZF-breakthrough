'''CZF Actor'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import platform
import tempfile
import zstandard as zstd

from czf.utils import get_zmq_dealer, RemoteModelManager
from czf.utils.model_saver import jit
from czf.pb import czf_pb2


class Actor:
    '''Actor'''
    def __init__(self, args, config, worker_manager):
        self._node = czf_pb2.Node(
            hostname=platform.node(),
            identity=f'actor-{args.suffix}',
        )

        # Actor mode
        self.algorithm = config['algorithm']
        assert self.algorithm in ('AlphaZero', 'MuZero')
        operation = {
            'AlphaZero': {
                None: czf_pb2.Job.Operation.ALPHAZERO_SEARCH,
                '1P': czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_1P,
                '2P': czf_pb2.Job.Operation.ALPHAZERO_EVALUATE_2P,
            },
            'MuZero': {
                None: czf_pb2.Job.Operation.MUZERO_SEARCH,
                '1P': czf_pb2.Job.Operation.MUZERO_EVALUATE_1P,
                '2P': czf_pb2.Job.Operation.MUZERO_EVALUATE_2P,
            },
        }
        self._operation = operation[self.algorithm][args.eval]
        if args.eval:
            assert args.eval in ('1P', '2P')
            print(f'[{self.algorithm} Evaluation {args.eval} Mode]', self._node.identity)
        else:
            print(f'[{self.algorithm} Training Mode]', self._node.identity)
        
        # worker manager
        self._worker_manager = worker_manager
        if self.algorithm == 'AlphaZero':
            self._worker_manager.run(
                num_cpu_worker=args.num_cpu_worker,
                num_gpu_worker=args.num_gpu_worker,
                num_gpu=args.num_gpu,
            )
        elif self.algorithm == 'MuZero':
            self._worker_manager.run(
                num_cpu_worker=args.num_cpu_worker,
                num_gpu_worker=args.num_gpu_worker,
                num_gpu_root_worker=args.num_gpu_root_worker,
                num_gpu=args.num_gpu,
            )

        # model
        self._model_info = czf_pb2.ModelInfo(name='default', version=-1)
        self._has_new_model = asyncio.Event()
        self._has_load_model = asyncio.Event()
        self._dctx = zstd.ZstdDecompressor()
        # connect to the remote model provider
        self._model_manager = RemoteModelManager(
            identity=f'model-manager-{args.suffix}',
            upstream=args.upstream,
            cache_size=8,
        )
        # connect to the broker
        self._broker = get_zmq_dealer(identity=self._node.identity,
                                      remote_address=args.broker)
        asyncio.create_task(self.__initialize())

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop(), self._dequeue_loop(),
                             self._load_model_loop())

    async def _recv_loop(self):
        '''a loop to receive `Packet` (`JobBatch`) from broker'''
        while True:
            raw = await self._broker.recv()
            # packet = czf_pb2.Packet.FromString(raw)
            # print(packet)
            await self.__on_job_batch(raw)

    async def _dequeue_loop(self):
        '''a loop to dequeue from WorkerManager and send `JobBatch`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while True:
            raw = await loop.run_in_executor(
                executor, self._worker_manager.wait_dequeue_result, 15)  #TODO
            if not raw:
                break
            await self._broker.send(raw)

    async def __on_job_batch(self, raw: bytes):
        '''enqueue jobs into WorkerManager and send `JobRequest`'''
        flush_job, name, version = self._worker_manager.enqueue_job_batch(raw)
        if flush_job != b'' or name != self._model_info.name or version > self._model_info.version:
            self._model_info.name = name
            self._model_info.version = version
            self._has_load_model.clear()
            self._has_new_model.set()
        if flush_job != b'':
            await self._has_load_model.wait()
            await self._broker.send(flush_job)
        # await self.__send_job_request() #TODO?

    async def _load_model_loop(self):
        '''a loop to load model'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while await self._has_new_model.wait():
            self._has_new_model.clear()
            model = await self._model_manager.get(self._model_info)
            await loop.run_in_executor(executor, self.__load_model, model)
            self._has_load_model.set()

    def __load_model(self, model: czf_pb2.Model):
        '''WorkerManager load model'''
        assert len(model.blobs) == 1
        print('load model', model.info.name, 'iteration', model.info.version)

        if self.algorithm == 'AlphaZero':
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(model.blobs[0])
                self._worker_manager.load_from_file(tmp_file.name)
        elif self.algorithm == 'MuZero':
            blob = model.blobs[0]
            model_blob = self._dctx.decompress(blob)
            # _, model_blob = jit(blob)
            self._worker_manager.load_from_bytes(model_blob)

    async def __initialize(self):
        '''initialize model and start to send job'''
        # load the latest model
        self._model_info.version = await self._model_manager.get_latest_version(
            self._model_info.name)
        model = await self._model_manager.get(self._model_info)
        self.__load_model(model)
        # start to send job
        await self.__send_job_request()

    async def __send_job_request(self, capacity=1280):
        '''helper to send a `JobRequest`'''
        packet = czf_pb2.Packet(job_request=czf_pb2.JobRequest(
            operation=self._operation, capacity=capacity))
        await self._broker.send(packet.SerializeToString())
