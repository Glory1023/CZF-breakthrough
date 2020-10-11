'''Helper functions to create a zmq socket'''
import zmq
import zmq.asyncio


def get_zmq_dealer(identity, remote_address):
    '''return a zmq `DEALER` socket'''
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.IDENTITY, identity)
    socket.connect(f'tcp://{remote_address}')
    return socket


def get_zmq_router(listen_port):
    '''return a zmq `ROUTER` socket'''
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f'tcp://*:{listen_port}')
    return socket
