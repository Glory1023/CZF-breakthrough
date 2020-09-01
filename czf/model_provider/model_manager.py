import zmq
import asyncio
import zmq.asyncio
from pathlib import Path

import czf_pb2
from .lru_cache import LRUCache


class LocalModelManager:
    def __init__(self, args):
        self.cache = LRUCache(capacity=args.cache_size)

        storage = Path(args.storage)
        if not storage.exists():
            storage.mkdir(parents=True)
        self.storage = storage

    async def get(self, model: czf_pb2.Model):
        key = (model.name, model.version)
        if key not in self.cache:
            model_dir = self.storage / model.name
            model_path = model_dir / f'{model.version}.pt'
            model.blobs[:] = [model_path.read_bytes()]
            self.cache[key] = model
        return self.cache[key]


class RemoteModelManager:
    def __init__(self, args):
        self.cache = LRUCache(capacity=args.cache_size)

        self.identity = f'model-provider-{args.suffix}'
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.IDENTITY, self.identity)
        socket.connect(f'tcp://{args.upstream}')
        self.upstream = socket

        asyncio.create_task(self.loop())

    async def loop(self):
        while True:
            raw = await self.upstream.recv()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_response':
                model = packet.model_response
                key = (model.name, model.version)
                if key not in self.cache:
                    self.cache[key] = asyncio.Future()
                self.cache[key].set_result(model)

    def get(self, model: czf_pb2.Model):
        key = (model.name, model.version)
        if key not in self.cache:
            self.cache[key] = asyncio.Future()
            asyncio.create_task(self.request_model(model))
        return self.cache[key]

    async def request_model(self, model: czf_pb2.Model):
        packet = czf_pb2.Packet()
        packet.model_request.CopyFrom(model)
        raw = packet.SerializeToString()
        await self.upstream.send(raw)
