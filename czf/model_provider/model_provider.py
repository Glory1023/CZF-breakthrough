'''CZF ModelProvider'''
import asyncio
from pathlib import Path

from czf.pb import czf_pb2
from czf.utils import get_zmq_router, LocalModelManager, RemoteModelManager


class ModelProvider:
    '''ModelProvider'''
    def __init__(self, args):
        self._provider = get_zmq_router(listen_port=args.listen)
        if args.storage:
            model_path = Path(args.storage) / 'model'
            self._model_manager = LocalModelManager(
                storage=model_path,
                cache_size=args.cache_size,
            )
        else:
            self._model_manager = RemoteModelManager(
                identity=f'model-provider-{args.suffix}',
                upstream=args.upstream,
                cache_size=args.cache_size,
            )

    async def loop(self):
        print("Model provider is running now ...")
        '''main loop'''
        while True:
            identity, raw = await self._provider.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            if packet_type == 'model_request':
                asyncio.create_task(self.__on_model_request(identity, packet.model_request))

    async def __on_model_request(self, identity: bytes, info: czf_pb2.ModelInfo):
        '''send `Model`'''
        if info.version == -1:
            version = self._model_manager.get_latest_version(info.name)
            info.version = version
        print("Recieve model request. Model version: ", info.version)
        model = self._model_manager.get(info)
        if model:
            packet = czf_pb2.Packet()
            packet.model_response.CopyFrom(model)
            self._provider.send_multipart([identity, packet.SerializeToString()])
            print("Send model successfully. Model version: ", info.version)
