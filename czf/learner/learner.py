'''CZF Learner'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from czf.learner.dataloader import PreprocessQueue
from czf.learner.trainer import TrainerRunner
from czf.pb import czf_pb2
from czf.utils import get_zmq_router, LocalModelManager


class Learner:
    '''Learner'''
    def __init__(self, args, config):
        # create directories
        storage_path = Path(args.storage_dir)
        checkpoint_path = storage_path / 'checkpoint'
        model_path = storage_path / 'model'
        log_path = storage_path / 'log'
        trajectory_path = storage_path / 'trajectory'
        all_path = (storage_path, checkpoint_path, model_path, log_path,
                    trajectory_path)
        for path in all_path:
            Path(path).mkdir(parents=True, exist_ok=True)
        # queue
        self._model_request = asyncio.Queue()
        self._trajectory = PreprocessQueue(
            num_player=config['game']['num_player'],
            num_action=config['game']['actions'],
            observation_config=config['game']['observation'],
            transform=config['learner'].get('transform'),
            r_heads=config['model']['r_heads'],
            v_heads=config['model']['v_heads'],
            kstep=config['learner']['rollout_steps'],
            nstep=config['mcts']['nstep'],
            discount_factor=config['mcts']['discount_factor'],
            num_proc=args.num_proc,
            use_prioritize=config['learner']['prioritized'],
        )
        # trainer
        self._trainer_runner = TrainerRunner(args, config, all_path,
                                             self._trajectory)
        # model provider
        self._model_peers = set()  # peers to receive the model
        self._model_provider = LocalModelManager(
            storage=model_path,
            cache_size=8,
        )
        self._socket = get_zmq_router(listen_port=args.listen)

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop(), self._model_request_loop(),
                             self._notify_model_loop())

    async def _recv_loop(self):
        '''receive loop'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'trajectory_batch':
                self._trajectory.put(raw)
            elif packet_type == 'model_request':
                await self._model_request.put((identity, packet.model_request))
            elif packet_type == 'evaluation_result':
                await self.__on_evaluation_result(packet.evaluation_result)
            elif packet_type == 'model_subscribe':
                self._model_peers.add(identity)
            elif packet_type == 'goodbye':
                self._model_peers.remove(identity)

    async def _notify_model_loop(self):
        '''loop to notify model update to peers'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while True:
            name, version = await loop.run_in_executor(
                executor, self._trainer_runner.get_notify)
            print('notify model', name, 'iteration', version)
            raw = czf_pb2.Packet(model_info=czf_pb2.ModelInfo(
                name=name,
                version=version,
            )).SerializeToString()
            for peer in self._model_peers:
                await self.__send_raw(peer, raw)

    async def _model_request_loop(self):
        '''send `Model` loop'''
        while True:
            identity, info = await self._model_request.get()
            if info.version == -1:
                version = self._model_provider.get_latest_version(info.name)
                info.version = version
            model = self._model_provider.get(info)
            packet = czf_pb2.Packet()
            packet.model_response.CopyFrom(model)
            await self.__send_packet(identity, packet)

    async def __on_evaluation_result(self, result: czf_pb2.EvaluationResult):
        '''store evaluation result'''
        step = result.iteration
        writer = self._trainer._summary_writer
        writer.add_scalar('eval/elo', result.elo, step)
        writer.add_scalar('eval/current_version', result.target.version, step)
        writer.add_scalar('eval/best_version', result.base.version, step)
        writer.add_scalars('eval/result', {
            'win': result.win,
            'draw': result.draw,
            'lose': result.lose,
        }, step)
        writer.flush()

    async def __send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self.__send_raw(identity, packet.SerializeToString())

    async def __send_raw(self, identity: bytes, raw: bytes):
        '''helper to send a zmq message'''
        await self._socket.send_multipart([identity, raw])
