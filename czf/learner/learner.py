'''CZF Learner'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from czf.learner.preprocessor.preprocessor import TrajectoryQueue
from czf.learner.trainer.trainer import TrainerRunner
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
        trajectory_path = storage_path / 'trajectory'
        all_path = (storage_path, checkpoint_path, model_path, log_path, trajectory_path)
        for path in all_path:
            Path(path).mkdir(parents=True, exist_ok=True)

        # preprocessor, trainer, and replay buffer
        algorithm = config['algorithm']
        if algorithm == 'AlphaZero':
            from czf.learner.preprocessor.alphazero_preprocessor import AlphaZeroPreprocessor
            from czf.learner.trainer.alphazero_trainer import AlphaZeroTrainer
            from czf.learner.replay_buffer.alphazero_replay_buffer import AlphaZeroReplayBuffer, AlphaZeroBatch
            trainer_cls = AlphaZeroTrainer
            dataloder_collate_fn = AlphaZeroBatch
            replay_buffer_cls = AlphaZeroReplayBuffer
            replay_buffer_exposed_methods = [
                '__len__',
                '__getitem__',
                'add_trajectory',
                'get_num_to_add',
                'get_states_to_train',
                'get_statistics',
                'reset_statistics',
                'save_trajectory',
                'get_weights',
            ]
            replay_buffer_config = dict(
                num_player=config['game']['num_player'],
                states_to_train=config['learner'].get('states_to_train', None),
                sequences_to_train=config['learner'].get('sequences_to_train', None),
                sample_ratio=config['learner'].get('sample_ratio', None),
                sample_states=config['learner'].get('sample_states', None),
                observation_config=config['game']['observation'],
                capacity=config['learner']['replay_buffer_size'],
                use_transformation=config['learner']['use_transformation'],
                game=config['game']['name'],
            )
            # preprocess queue
            self._trajectory_queue = TrajectoryQueue(preprocessor_cls=AlphaZeroPreprocessor,
                                                     num_proc=args.num_proc)
        elif algorithm == 'MuZero':
            from czf.learner.preprocessor.muzero_preprocessor import MuZeroPreprocessor
            from czf.learner.trainer.muzero_trainer import MuZeroTrainer
            from czf.learner.trainer.stochastic_muzero_trainer import StochasticMuZeroTrainer
            from czf.learner.replay_buffer.muzero_replay_buffer import MuZeroReplayBuffer, MuZeroRolloutBatch
            is_stochastic = config['game']['is_stochastic']
            trainer_cls = StochasticMuZeroTrainer if is_stochastic else MuZeroTrainer
            dataloder_collate_fn = MuZeroRolloutBatch
            replay_buffer_cls = MuZeroReplayBuffer
            replay_buffer_exposed_methods = [
                '__len__',
                '__getitem__',
                'add_trajectory',
                'get_num_to_add',
                'get_states_to_train',
                'get_statistics',
                'reset_statistics',
                'save_trajectory',
                'restore_trajectory',
                'get_weights',
                'get_mean_weight',
                'update_weights',
                'copy_weights',
                'write_back_weights',
            ]
            replay_buffer_config = dict(
                num_player=config['game']['num_player'],
                states_to_train=config['learner'].get('states_to_train', None),
                sequences_to_train=config['learner'].get('sequences_to_train', None),
                sample_ratio=config['learner'].get('sample_ratio', None),
                sample_states=config['learner'].get('sample_states', None),
                observation_config=config['game']['observation'],
                kstep=config['learner']['rollout_steps'],
                capacity=config['learner']['replay_buffer_size'],
                is_stochastic=config['game']['is_stochastic'],
            )
            # preprocess queue
            self._trajectory_queue = TrajectoryQueue(
                preprocessor_cls=MuZeroPreprocessor,
                num_proc=args.num_proc,
                num_player=config['game']['num_player'],
                transform=config['learner'].get('transform'),
                r_heads=config['model']['r_heads'],
                v_heads=config['model']['v_heads'],
                kstep=config['learner']['rollout_steps'],
                nstep=config['mcts']['nstep'],
                discount_factor=config['mcts']['discount_factor'],
                use_prioritize=config['learner']['prioritized'],
                is_stochastic=config['game']['is_stochastic'],
            )

        # trainer
        self._trainer_runner = TrainerRunner(
            args,
            config,
            all_path,
            self._trajectory_queue,
            trainer_cls,
            dataloder_collate_fn,
            replay_buffer_cls,
            replay_buffer_exposed_methods,
            replay_buffer_config,
        )
        # model provider
        self._model_request = asyncio.Queue()
        self._model_peers = set()  # peers to receive the model
        self._model_provider = LocalModelManager(
            storage=model_path,
            cache_size=8,
        )
        self._socket = get_zmq_router(listen_port=args.listen)

    async def loop(self):
        '''main loop'''
        await asyncio.gather(
            self._recv_loop(),
            self._model_request_loop(),
            self._notify_model_loop(),
        )

    async def _recv_loop(self):
        '''receive loop'''
        while True:
            identity, raw = await self._socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'trajectory_batch':
                self._trajectory_queue.put(raw)
            elif packet_type == 'model_request':
                await self._model_request.put((identity, packet.model_request))
            elif packet_type == 'model_subscribe':
                self._model_peers.add(identity)
            elif packet_type == 'goodbye':
                self._model_peers.remove(identity)
                print("Game-server leave. (identity: ", identity, ")")

    async def _notify_model_loop(self):
        '''loop to notify model update to peers'''
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        while True:
            name, version = await loop.run_in_executor(executor, self._trainer_runner.get_notify)
            print('notify model', name, 'iteration', version)
            raw = czf_pb2.Packet(model_info=czf_pb2.ModelInfo(
                name=name,
                version=version,
            )).SerializeToString()
            for peer in self._model_peers:
                await self._send_raw(peer, raw)

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
            await self._send_packet(identity, packet)

    async def _send_packet(self, identity: bytes, packet: czf_pb2.Packet):
        '''helper to send a `Packet` to `identity`'''
        await self._send_raw(identity, packet.SerializeToString())

    async def _send_raw(self, identity: bytes, raw: bytes):
        '''helper to send a zmq message'''
        await self._socket.send_multipart([identity, raw])
