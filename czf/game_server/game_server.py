'''CZF Game Server'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ctypes
import multiprocessing as mp
import platform

from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer, Queue
from czf.game_server.env_manager import EnvManager


async def run_env_manager(*args):
    '''run :class:`EnvManager`'''
    manager = EnvManager(*args)
    manager.start_search()
    await manager.loop()


class ModelInfo(ctypes.Structure):
    '''ModelInfo shared between processes'''
    _fields_ = [('name', ctypes.c_wchar_p), ('version', ctypes.c_int)]


class GameServer:
    '''Game Server'''
    def __init__(self, args, config, callbacks):
        self._node = czf_pb2.Node(identity=f'game-server-{args.suffix}',
                                  hostname=platform.node())
        # model
        self._model_info = mp.Value(ModelInfo, 'default', -1)
        self._has_new_model = asyncio.Event()
        # server mode
        algorithm = config['algorithm']
        assert algorithm in ('AlphaZero', 'MuZero')
        if args.eval:
            print(f'[{algorithm} Evaluation Mode]', self._node.identity)
        else:
            print(f'[{algorithm} Training Mode]', self._node.identity)

        # start EnvManager
        self._num_env = args.num_env
        self._job_queue = Queue()
        self._trajectory_queue = Queue()
        self._pipe = [mp.Pipe() for i in range(args.num_proc)]
        self._manager = [
            mp.Process(
                target=lambda *args: asyncio.run(run_env_manager(*args)),
                args=(
                    args,
                    config,
                    callbacks,
                    index,
                    self._model_info,
                    pipe,
                    self._job_queue,
                    self._trajectory_queue,
                )) for index, (_, pipe) in enumerate(self._pipe)
        ]
        for manager in self._manager:
            manager.start()

        # trajectory upstream
        print('connect to learner @', args.upstream)
        self._upstream = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.upstream,
        )
        asyncio.create_task(self.__send_model_subscribe())
        # connect to broker
        print('connect to broker  @', args.broker)
        self._broker = get_zmq_dealer(
            identity=self._node.identity,
            remote_address=args.broker,
        )

    def terminate(self):
        '''terminate all EnvManager'''
        for manager in self._manager:
            manager.join()

    async def loop(self):
        '''main loop'''
        await asyncio.gather(
            self._recv_job_loop(),
            self._recv_model_info_loop(),
            self._send_job_loop(),
            self._send_trajectory_loop(),
        )

    async def _recv_job_loop(self):
        '''a loop to receive `Job`'''
        while True:
            raw = await self._broker.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            #print(packet)
            if packet_type == 'job':
                job = packet.job
                index = job.payload.env_index // self._num_env
                self._pipe[index][0].send(job.SerializeToString())
            elif packet_type == 'job_batch':
                jobs = packet.job_batch.jobs
                for job in jobs:
                    index = job.payload.env_index // self._num_env
                    self._pipe[index][0].send(job.SerializeToString())

    async def _recv_model_info_loop(self):
        '''a loop to receive `ModelInfo`'''
        while True:
            raw = await self._upstream.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_info':
                version = packet.model_info.version
                # assert packet._model_info.name == self._model_info.name
                if version > self._model_info.version:
                    self._model_info.version = version
                    self._has_new_model.set()

    async def _send_job_loop(self):
        '''a loop to send `Job`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._job_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            await self._broker.send(raw)

    async def _send_trajectory_loop(self):
        '''a loop to send `Trajectory`'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        queue = self._trajectory_queue
        while True:
            raw = await loop.run_in_executor(executor, queue.get)
            await self._upstream.send(raw)

    async def __send_model_subscribe(self):
        '''helper to send a `model_subscribe` to optimizer'''
        packet = czf_pb2.Packet(model_subscribe=czf_pb2.Heartbeat())
        await self._upstream.send(packet.SerializeToString())
