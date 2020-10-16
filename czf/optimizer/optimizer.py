'''CZF Optimizer'''
import asyncio
from pathlib import Path

from czf.optimizer.trainer import Trainer
from czf.optimizer.dataloader import ReplayBuffer
from czf.pb import czf_pb2
from czf.utils import get_zmq_router, LocalModelManager


class Optimizer:
    '''Optimizer'''
    def __init__(self, args, config):
        # create directories
        storage_path = Path(args.storage_dir)
        model_path = storage_path / config['optimizer'].get(
            'model_path', 'model')
        traced_model_path = storage_path / config['optimizer'].get(
            'traced_model_path', 'traced_model')
        trajectory_path = storage_path / config['optimizer'].get(
            'trajectory_path', 'trajectory')
        for path in (model_path, traced_model_path, trajectory_path):
            Path(path).mkdir(parents=True, exist_ok=True)
        # model provider
        self._model_peers = set()  # peers to receive the model
        self._model_provider = LocalModelManager(
            storage=traced_model_path,
            cache_size=8,
        )
        self._socket = get_zmq_router(listen_port=args.listen)
        # replay buffer
        self._replay_buffer = ReplayBuffer(
            num_player=config['game']['num_player'],
            observation_shape=config['game']['observation_shape'],
            kstep=config['optimizer']['rollout_steps'],
            nstep=config['mcts']['nstep'],
            discount_factor=config['mcts']['discount_factor'],
            capacity=config['optimizer']['replay_buffer_size'],
            train_freq=config['optimizer']['frequency'],
        )
        # trainer
        self._trainer = Trainer(args, config, model_path, traced_model_path)
        self._trainer.save_model(trace=True)

    async def loop(self):
        '''main loop'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            self._model_peers.add(identity)
            if packet_type == 'model_request':
                asyncio.create_task(
                    self.__on_model_request(identity, packet.model_request))
            elif packet_type == 'trajectory_batch':
                asyncio.create_task(
                    self.__on_recv_trajectory(packet.trajectory_batch))

    async def __on_model_request(self, identity: str,
                                 model: czf_pb2.ModelInfo):
        '''send `Model`'''
        if model.version == -1:
            version = self._model_provider.get_latest_version(model.name)
            model.version = version
        response_model = self._model_provider.get(model)
        await self.__send_packet(identity,
                                 czf_pb2.Packet(model_response=response_model))

    async def __on_recv_trajectory(self,
                                   trajectory_batch: czf_pb2.TrajectoryBatch):
        '''store trajectory'''
        for trajectory in trajectory_batch.trajectories:
            self._replay_buffer.add_trajectory(trajectory)
        if self._replay_buffer.is_ready():
            self._trainer.train(self._replay_buffer)
            self._trainer.save_model()
            await self.__notify_model_update(self._trainer.model_name,
                                             self._trainer.model_version)

    async def __notify_model_update(self, name, version):
        '''notify model update to peers'''
        raw = czf_pb2.Packet(model_info=czf_pb2.ModelInfo(
            name=name,
            version=version,
        )).SerializeToString()
        for peer in self._model_peers:
            await self.__send_raw(peer, raw)

    async def __send_packet(self, identity: str, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self.__send_raw(identity, packet.SerializeToString())

    async def __send_raw(self, identity: str, raw):
        '''helper to send a zmq message'''
        await self._socket.send_multipart([identity, raw])
