'''CZF Optimizer'''
import asyncio
from pathlib import Path

from czf.optimizer.trainer import Trainer
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
        for p in (model_path, traced_model_path, trajectory_path):
            Path(p).mkdir(parents=True, exist_ok=True)
        # model provider
        self.model_peers = set()  # peers to receive the model
        self.model_provider = LocalModelManager(
            storage=traced_model_path,
            cache_size=8,
        )
        self.socket = get_zmq_router(listen_port=args.listen)
        self.trainer = Trainer(args, config, model_path, traced_model_path)
        self.trainer.save_model(trace=True)

    async def loop(self):
        '''main loop'''
        while True:
            identity, raw = await self.socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'model_request':
                asyncio.create_task(
                    self.__on_model_request(identity, packet.model_request))
            elif packet_type == 'trajectory':
                asyncio.create_task(
                    self.__on_recv_trajectory(packet.trajectory))

    async def __on_model_request(self, identity: str, model: czf_pb2.Model):
        '''send `Model`'''
        self.model_peers.add(identity)
        if model.version == -1:
            version = self.model_provider.get_latest_version(model.name)
            model.version = version
        response_model = self.model_provider.get(model)
        await self.__send_packet(identity,
                                 czf_pb2.Packet(model_response=response_model))

    async def __on_recv_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''store trajectory'''
        return
        # self.replay_buffer.add_trajectory(packet.trajectory)
        # if self.replay_buffer.ready:
        #     self.trainer.train()
        #     self.replay_buffer.ready = False
        #     self.model_version += 1
        #     self.trainer.save_model()
        #     await self.__notify_model_update()

    async def __notify_model_update(self, name, version):
        '''notify model update to peers'''
        raw = czf_pb2.Packet(model=czf_pb2.Model(
            name=name,
            version=version,
        )).SerializeToString()
        for peer in self.model_peers:
            await self.__send_raw(peer, raw)

    async def __send_packet(self, identity: str, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self.__send_raw(identity, packet.SerializeToString())

    async def __send_raw(self, identity: str, raw):
        '''helper to send a zmq message'''
        await self.socket.send_multipart([identity, raw])
