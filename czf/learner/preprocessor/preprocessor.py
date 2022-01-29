'''CZF Preprocessor'''
import abc
from multiprocessing import Queue
import torch
import torch.multiprocessing as mp

from czf.pb import czf_pb2


def run_preprocessor(preprocessor_cls, raw_queue, result_queue, **kwargs):
    '''run :class:`Preprocessor`'''
    preprocessor = preprocessor_cls(result_queue, **kwargs)
    while True:
        raw = raw_queue.get()
        packet = czf_pb2.Packet.FromString(raw)
        batch = packet.trajectory_batch.trajectories
        for trajectory in batch:
            preprocessor.add_trajectory(trajectory)
        packet.ClearField('trajectory_batch')


class TrajectoryQueue:
    '''Preprocess job queue'''
    def __init__(self, preprocessor_cls, num_proc, **kwargs):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # multiprocessing
        self._raw_queue = Queue()
        self._result_queue = Queue()
        self._process = [
            mp.Process(target=run_preprocessor,
                       args=(
                           preprocessor_cls,
                           self._raw_queue,
                           self._result_queue,
                       ),
                       kwargs=kwargs) for _ in range(num_proc)
        ]
        for process in self._process:
            process.start()

    def put(self, raw: bytes):
        '''Enqueue raw protobuf packet'''
        self._raw_queue.put(raw)

    def get(self):
        '''Dequeue preprocessed trajectory'''
        return self._result_queue.get()

    def get_all(self):
        '''Dequeue all preprocessed trajectory'''
        x = self._result_queue.get()
        qsize = self._result_queue.qsize()
        return [x] + [self._result_queue.get() for _ in range(qsize)]


class Preprocessor(abc.ABC):
    '''Preprocess raw packet to trajectories'''
    @abc.abstractmethod
    def add_trajectory(self, trajectory: czf_pb2.Trajectory):
        '''Add a trajectory to the replay buffer

        Args:
            trajectory: czf_pb2.Trajectory contains one trajectory

        Raises:
            NotImplementedError: Preprocessor must be able to add trajectory
        '''
        raise NotImplementedError('Preprocessor must be able to add trajectory')
