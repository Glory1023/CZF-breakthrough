'''CZF Model Manager'''
import asyncio
from pathlib import Path
import os

from czf.pb import czf_pb2
from czf.utils import LRUCache, get_zmq_dealer


class LocalModelManager:
    '''Local Model Manager'''
    __slots__ = ['storage', '_cache']

    def __init__(self, storage, cache_size):
        self._cache = LRUCache(capacity=cache_size)
        self.storage = Path(storage)
        self.storage.mkdir(parents=True, exist_ok=True)

    def get(self, model: czf_pb2.ModelInfo) -> czf_pb2.Model:
        '''get the `Model` from storage'''
        key = (model.name, model.version)
        if key not in self._cache:
            model_dir = self.storage / model.name
            model_path = model_dir / f'{model.version:05d}.pt'
            self._cache[key] = czf_pb2.Model(
                info=model,
                blobs=[model_path.read_bytes()],
            )
        return self._cache[key]

    def get_latest_version(self, name: str) -> int:
        '''get the latest model version by `name`'''
        model_dir = self.storage / name
        model_file = os.path.realpath(model_dir / 'latest.pt')
        return int(os.path.basename(model_file)[:-3])


class RemoteModelManager:
    '''Remote Model Manager'''
    __slots__ = ['has_new_model', 'upstream', '_cache', '_latest_version']

    def __init__(self, identity, upstream, cache_size):
        self._cache = LRUCache(capacity=cache_size)
        self._latest_version = {}
        self.upstream = get_zmq_dealer(identity=identity,
                                       remote_address=upstream)
        self.has_new_model = asyncio.Event()
        asyncio.create_task(self._recv_loop())

    def get(self, model: czf_pb2.ModelInfo) -> czf_pb2.Model:
        '''get the `Model` from upstream'''
        key = (model.name, model.version)
        if key not in self._cache:
            self._cache[key] = asyncio.Future()
            asyncio.create_task(self.__send_model_request(model))
        return self._cache[key]

    async def get_latest_version(self, name: str) -> int:
        '''get the latest model version by `name`'''
        if name not in self._latest_version:
            self._latest_version[name] = asyncio.Future()
            await self.__send_model_request(
                czf_pb2.ModelInfo(
                    name=name,
                    version=-1,
                ))
            return await self._latest_version[name]
        return self._latest_version[name]

    async def _recv_loop(self):
        '''a loop to receive `Model`'''
        while True:
            packet = await self.__recv_packet()
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_response':
                model = packet.model_response
                name, version = model.info.name, model.info.version
                latest_version = self._latest_version.get(name, -1)
                if asyncio.isfuture(latest_version):
                    self._latest_version[name].set_result(version)
                elif version > latest_version:
                    self._latest_version[name] = version
                key = (name, version)
                if key not in self._cache:
                    self._cache[key] = asyncio.Future()
                self._cache[key].set_result(model)
            elif packet_type == 'model_info':
                model = packet.model_info
                latest_version = self._latest_version.get(name)
                if isinstance(latest_version,
                              int) and model.version > latest_version:
                    self.has_new_model.set()
                self._latest_version[model.name] = model.version

    async def __send_model_request(self, model: czf_pb2.ModelInfo):
        '''helper to send a `Model`'''
        packet = czf_pb2.Packet(model_request=model)
        await self.__send_packet(packet)

    async def __send_packet(self, packet: czf_pb2.Packet):
        '''helper to send a `Packet`'''
        raw = packet.SerializeToString()
        await self.upstream.send(raw)

    async def __recv_packet(self) -> czf_pb2.Packet:
        '''helper to receive a `Packet`'''
        raw = await self.upstream.recv()
        packet = czf_pb2.Packet.FromString(raw)
        return packet
