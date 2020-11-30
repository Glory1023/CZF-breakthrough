'''CZF NamedBroker'''
from czf.utils import get_zmq_router


class NamedBroker:
    '''NamedBroker'''
    def __init__(self, args):
        self._next = {
            args.a.encode(): args.g.encode(),
            args.g.encode(): args.a.encode(),
        }
        self._socket = get_zmq_router(listen_port=args.listen)

    async def loop(self):
        '''main loop'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            await self._socket.send_multipart([self._next[identity], raw])
