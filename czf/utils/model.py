'''CZF Model Manager'''
import asyncio
from pathlib import Path
import os

from czf.pb import czf_pb2
from czf.utils import LRUCache, get_zmq_dealer


class LocalModelManager:
    '''Local Model Manager'''
    __slots__ = ['_storage', '_cache']

    def __init__(self, storage: str, cache_size: int):
        self._cache = LRUCache(capacity=cache_size)
        self._storage = Path(storage)
        self._storage.mkdir(parents=True, exist_ok=True)

    def get(self, info: czf_pb2.ModelInfo) -> czf_pb2.Model:
        '''get the `Model` from storage'''
        key = (info.name, info.version)
        if key not in self._cache:
            model_path = self._storage / info.name / f'{info.version:05d}.pt'
            model = czf_pb2.Model()
            model.info.CopyFrom(info)
            model.blobs.append(model_path.read_bytes())
            self._cache[key] = model
        return self._cache[key]

    def get_latest_version(self, name: str) -> int:
        '''get the latest model version by `name`'''
        model_dir = self._storage / name
        model_file = os.path.realpath(model_dir / 'latest.pt')
        return int(os.path.basename(model_file)[:-3])


class LocalModelManagerAsync(LocalModelManager):
    '''Local Model Manager with asyncio'''
    def __init__(self, identity, upstream, cache_size, storage):
        super().__init__(storage, cache_size)
        # {model_name (`str`): latest_version (`int`) }
        self._latest_version = {}
        self.has_new_model = asyncio.Event()
        self.upstream = get_zmq_dealer(identity=identity,
                                       remote_address=upstream)
        asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        '''a loop to receive `ModelInfo`'''
        while True:
            packet = await self.__recv_packet()
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_info':
                name = packet.model_info.name
                latest_version = self.get_latest_version(name)
                if latest_version > self._latest_version[name]:
                    self._latest_version[name] = latest_version
                    self.has_new_model.set()

    async def __recv_packet(self) -> czf_pb2.Packet:
        '''helper to receive a `Packet`'''
        raw = await self.upstream.recv()
        packet = czf_pb2.Packet.FromString(raw)
        return packet


class RemoteModelManager:
    '''Remote Model Manager'''
    __slots__ = ['has_new_model', 'upstream', '_cache', '_latest_version']

    def __init__(self, identity, upstream, cache_size):
        # {(name, version): model_blob (`czf_pb2.Model` or `asyncio.Future`)}
        self._cache = LRUCache(capacity=cache_size)
        # {model_name (`str`): latest_version (`int` or `asyncio.Future`) }
        self._latest_version = {}
        self.has_new_model = asyncio.Event()
        self.upstream = get_zmq_dealer(identity=identity,
                                       remote_address=upstream)
        asyncio.create_task(self._recv_loop())

    def get(self, info: czf_pb2.ModelInfo) -> asyncio.Future:
        '''get the `Model` from cache or upstream'''
        key = (info.name, info.version)
        if key in self._cache:
            result = asyncio.Future()
            result.set_result(self._cache[key])
            return result
        self._cache[key] = asyncio.Future()
        asyncio.create_task(self.__send_model_request(info))
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
        '''a loop to receive `ModelInfo` and `Model`'''
        while True:
            packet = await self.__recv_packet()
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_response':
                model = packet.model_response
                info = model.info
            elif packet_type == 'model_info':
                model, info = None, packet.model_info
            name, version = info.name, info.version
            # update latest version
            latest_version = self._latest_version.get(name, -1)
            if asyncio.isfuture(latest_version):  # get_latest_version
                self._latest_version[name].set_result(version)
                self._latest_version[name] = version
            elif version > latest_version:
                self._latest_version[name] = version
                self.has_new_model.set()
            # update model
            if model:
                key = (name, version)
                if asyncio.isfuture(self._cache.get(key)):  # get
                    self._cache[key].set_result(model)
                self._cache[key] = model

    async def __send_model_request(self, info: czf_pb2.ModelInfo):
        '''helper to send a `Model`'''
        packet = czf_pb2.Packet(model_request=info)
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
