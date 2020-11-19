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
        # queue
        self._trajectory = asyncio.Queue()
        self._model_request = asyncio.Queue()
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
        # restore the replay buffer
        if args.restore_buffer:
            pass  # TODO
        # trainer
        self._checkpoint_freq = config['learner']['checkpoint_freq']
        self._trainer = Trainer(config, checkpoint_path, model_path, log_path,
                                args.model_name, args.restore)
        self._trainer.save_model()
        # pretrain the trajectory
        if args.pretrain_trajectory:
            pass  # TODO

    async def loop(self):
        '''main loop'''
        await asyncio.gather(self._recv_loop(), self._model_request_loop(),
                             self._trajectory_loop())

    async def _recv_loop(self):
        '''receive loop'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'trajectory_batch':
                await self._trajectory.put(packet.trajectory_batch)
            elif packet_type == 'model_request':
                await self._model_request.put((identity, packet.model_request))
            elif packet_type == 'evaluation_result':
                await self.__on_evaluation_result(packet.evaluation_result)
            elif packet_type == 'model_subscribe':
                self._model_peers.add(identity)
            elif packet_type == 'goodbye':
                self._model_peers.remove(identity)

    async def _trajectory_loop(self):
        '''store trajectory loop'''
        while True:
            trajectory_batch = await self._trajectory.get()
            for trajectory in trajectory_batch.trajectories:
                self._replay_buffer.add_trajectory(trajectory)
            if self._replay_buffer.is_ready():
                self._trainer.log_statistics(self._replay_buffer)
                self._replay_buffer.save_trajectory(self._trajectory_path,
                                                    self._trainer.iteration)
                self._trainer.train(self._replay_buffer)
                save_ckpt = (self._trainer.iteration %
                             self._checkpoint_freq == 0)
                self._trainer.save_model(save_ckpt)
                await self.__notify_model_update(self._trainer.model_name,
                                                 self._trainer.iteration)

    async def _model_request_loop(self):
        '''send `Model` loop'''
        while True:
            identity, info = await self._model_request.get()
            if info.version == -1:
                version = self._model_provider.get_latest_version(info.name)
                info.version = version
            model = self._model_provider.get(info)
            packet = czf_pb2.Packet()
            packet.model_response.CopyFrom(model)
            await self.__send_packet(identity, packet)

    async def __on_evaluation_result(self, result: czf_pb2.EvaluationResult):
        '''store evaluation result'''
        step = result.iteration
        writer = self._trainer._summary_writer
        writer.add_scalar('eval/elo', result.elo, step)
        writer.add_scalar('eval/current_version', result.target.version, step)
        writer.add_scalar('eval/best_version', result.base.version, step)
        writer.add_scalars('eval/result', {
            'win': result.win,
            'draw': result.draw,
            'lose': result.lose,
        }, step)
        writer.flush()

    async def __notify_model_update(self, name: str, version: str):
        '''notify model update to peers'''
        print('notify model', name, 'iteration', version)
        raw = czf_pb2.Packet(model_info=czf_pb2.ModelInfo(
            name=name,
            version=version,
        )).SerializeToString()
        for peer in self._model_peers:
            await self.__send_raw(peer, raw)

    async def __send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self.__send_raw(identity, packet.SerializeToString())

    async def __send_raw(self, identity: bytes, raw: bytes):
        '''helper to send a zmq message'''
        await self._socket.send_multipart([identity, raw])
