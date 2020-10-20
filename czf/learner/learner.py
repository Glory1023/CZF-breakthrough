'''CZF Learner'''
import asyncio
from pathlib import Path

from czf.learner.trainer import Trainer
from czf.learner.dataloader import ReplayBuffer
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
        self._trajectory_path = storage_path / 'trajectory'
        for path in (checkpoint_path, model_path, log_path,
                     self._trajectory_path):
            Path(path).mkdir(parents=True, exist_ok=True)
        # model provider
        self._model_peers = set()  # peers to receive the model
        self._model_provider = LocalModelManager(
            storage=model_path,
            cache_size=8,
        )
        self._socket = get_zmq_router(listen_port=args.listen)
        # replay buffer
        self._replay_buffer = ReplayBuffer(
            num_player=config['game']['num_player'],
            observation_shape=config['game']['observation_shape'],
            kstep=config['learner']['rollout_steps'],
            nstep=config['mcts']['nstep'],
            discount_factor=config['mcts']['discount_factor'],
            capacity=config['learner']['replay_buffer_size'],
            train_freq=config['learner']['frequency'],
        )
        # trainer
        self._trainer = Trainer(config, checkpoint_path, model_path, log_path,
                                args.restore)
        self._checkpoint_freq = config['learner']['checkpoint_freq']
        self._trainer.save_model()

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
            self._replay_buffer.save_trajectory(self._trajectory_path,
                                                self._trainer.iteration)
            self._trainer.train(self._replay_buffer)
            save_ckpt = (self._trainer.iteration % self._checkpoint_freq == 0)
            self._trainer.save_model(save_ckpt)
            await self.__notify_model_update(self._trainer.model_name,
                                             self._trainer.iteration)

    async def __notify_model_update(self, name, version):
        '''notify model update to peers'''
        print('notify model', name, 'iteration', version)
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
