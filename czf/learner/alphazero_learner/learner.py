#! /usr/bin/env python3
import argparse
import asyncio
import random
import tempfile
import yaml
import zmq
import zmq.asyncio
from pathlib import Path
import zstandard as zstd

import numpy as np
from collections import Counter
import subprocess
import os
import psutil

import torch
import torch.cuda
import torch.jit
import torch.nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from czf.env import czf_env
from czf.pb import czf_pb2
from czf.learner.nn import AlphaZero
from czf.utils import get_zmq_router, LRUCache


class ReplayBuffer(Dataset):
    def __init__(self, game, capacity, train_freq):
        self.game = game
        self.capacity = capacity
        self.train_freq = train_freq

        self.data = []
        self.num_new_states = 0
        self.ready = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_trajectory(self, trajectory):
        state = self.game.new_initial_state()
        for pb_state in trajectory.states:
            observation_tensor = torch.tensor(state.observation_tensor).view(self.game.observation_tensor_shape)
            policy = torch.tensor(pb_state.evaluation.policy)
            returns = torch.tensor(trajectory.statistics.rewards)

            self.data.append((observation_tensor, policy, returns))

            self.num_new_states += 1
            state.apply_action(pb_state.transition.action)

            if len(self.data) > self.capacity:
                self.data.pop(0)

        if self.num_new_states >= self.train_freq:
            self.ready = True
            self.num_new_states -= self.train_freq


class Learner:
    def __init__(self, args, config):
        self.compressor = zstd.ZstdCompressor(level=config['misc']['compression_level'], threads=-1)
        self.decompressor = zstd.ZstdDecompressor()

        self.storage_dir = Path(args.storage_dir)
        self.model_dir = self.storage_dir / 'model'
        self.trajectory_dir = self.storage_dir / 'trajectory'
        self.checkpoint_dir = self.storage_dir / 'checkpoint'
        self.log_dir = self.storage_dir / 'log'

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # self.model_response_cache[iteration]
        self.model_response_cache = LRUCache(capacity=4)
        self.iteration = 0
        self.trajectories = 0

        self.pb_trajectory_batch = czf_pb2.TrajectoryBatch()

        self.game = czf_env.load_game(config['game']['name'])
        self.replay_buffer_reuse = config['learner']['replay_buffer_reuse']
        self.replay_retention = config['learner']['replay_buffer_size'] / config['learner']['frequency']

        self.checkpoint_freq = config['learner']['checkpoint_freq']

        self.peers = set()

        self.replay_buffer = ReplayBuffer(
            self.game,
            config['learner']['replay_buffer_size'],
            config['learner']['frequency']
        )

        self.batch_size = config['learner']['batch_size']
        self.use_transformation = config['learner']['use_transformation']

        self.model_config = config['model']

        self.socket = get_zmq_router(listen_port=args.listen)

        num_gpu = torch.cuda.device_count()
        self.gpus = [i for i in range(num_gpu)]
        self.device = torch.device(f'cuda:{self.gpus[0]}')

        latest_checkpoint = torch.load(self.checkpoint_dir / args.checkpoint) if args.restore else None

        self._model_kwargs = dict(
            observation_tensor_shape=self.game.observation_tensor_shape,
            action_dim=self.game.num_distinct_actions,
            channels=self.model_config['channels'],
            blocks=self.model_config['blocks'],
            v_heads=self.game.num_players,
            backbone=self.model_config['backbone'],
            fc_hidden_dimension=self.model_config.get('fc_hidden_dimension', 16)
        )
        self.model = AlphaZero(**self._model_kwargs)

        if latest_checkpoint:
            self.model.load_state_dict(latest_checkpoint['model'])
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config['learner']['learning_rate']['initial'],
            momentum=config['learner']['momentum'],
            weight_decay=config['learner']['weight_decay'],
            nesterov=True
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['learner']['learning_rate']['milestones'],
            gamma=config['learner']['learning_rate']['gamma']
        )

        if latest_checkpoint:
            self.iteration = latest_checkpoint['iteration']
            self.trajectories = latest_checkpoint['trajectories']
            if not config['learner']['learning_rate']['reset_lr']:
                print('[Learning Rate] Restore scheduler')
                self.optimizer.load_state_dict(latest_checkpoint['optimizer'])
                self.scheduler.load_state_dict(latest_checkpoint['scheduler'])
            self.restore_replay_buffer()
        else:
            self.save_model()
            self.save_checkpoint()
        self.summary_writer = SummaryWriter(log_dir=self.log_dir, purge_step=self.iteration)

        if args.pretrain_trajectory_dir:
            print('pretrain trajectory directory:', args.pretrain_trajectory_dir)
            self.pretrain_trajectory(args.pretrain_trajectory_dir)

    async def loop(self):
        while True:
            identity, raw = await self.socket.recv_multipart()
            packet = czf_pb2.Packet.FromString(raw)
            packet_type = packet.WhichOneof('payload')
            # print(packet)
            if packet_type == 'model_subscribe':
                self.peers.add(identity)
            elif packet_type == 'goodbye':
                self.peers.remove(identity)
            elif packet_type == 'model_request':
                await self.send_model(identity, packet.model_request)
            elif packet_type == 'trajectory_batch':
                for trajectory in packet.trajectory_batch.trajectories:
                    self.add_trajectory(trajectory)
                    if self.replay_buffer.ready:
                        self.replay_buffer.ready = False
                        self.train()

                        process = psutil.Process()
                        process_memory = process.memory_info()
                        for name in process_memory._fields:
                            value = getattr(process_memory, name)
                            self.summary_writer.add_scalar('Memory/{}'.format(name.capitalize()), value, self.iteration)

                        print('Iteration {} : Replay buffer {}'.format(self.iteration, len(self.replay_buffer)))

                        self.iteration += 1
                        self.scheduler.step()
                        self.save_model()
                        if self.iteration % self.checkpoint_freq == 0:
                            self.save_checkpoint()
                        await self.notify_update()

    def restore_replay_buffer(self):
        need_iteration = (self.replay_buffer.capacity // self.replay_buffer.train_freq) + 1
        print('self.iteration :', self.iteration)
        print('need iteration :', need_iteration)
        for iteration in range(max(0, self.iteration - need_iteration), self.iteration):
            # print('Iteration :', iteration)
            trajectory_path = self.trajectory_dir / f'{iteration:05d}.pb.zstd'
            compressed_trajectory = trajectory_path.read_bytes()
            decompressed_trajectory = self.decompressor.decompress(compressed_trajectory)
            pb_trajectory_batch = czf_pb2.TrajectoryBatch()
            pb_trajectory_batch.ParseFromString(decompressed_trajectory)
            # print(pb_trajectory_batch)
            for trajectory in pb_trajectory_batch.trajectories:
                self.replay_buffer.add_trajectory(trajectory)
            print('Replay Buffer {}: '.format(iteration), len(self.replay_buffer))
        self.replay_buffer.num_new_states = 0
        self.replay_buffer.ready = False
        print('Finish restoring replay buffer')

    def pretrain_trajectory(self, pretrain_trajectory_dir):
        for file_path in sorted(Path(pretrain_trajectory_dir).glob('*.pb.zstd')):
            print(file_path)
            compressed_trajectory = file_path.read_bytes()
            decompressed_trajectory = self.decompressor.decompress(compressed_trajectory)
            pb_trajectory_batch = czf_pb2.TrajectoryBatch()
            pb_trajectory_batch.ParseFromString(decompressed_trajectory)
            # print(pb_trajectory_batch)
            for trajectory in pb_trajectory_batch.trajectories:
                self.add_trajectory(trajectory)
                if self.replay_buffer.ready:
                    self.replay_buffer.ready = False
                    self.train()
                    print('Iteration {}: Replay buffer {}'.format(self.iteration, len(self.replay_buffer)))
                    self.iteration += 1
                    self.scheduler.step()
                    self.save_model()
                    if self.iteration % self.checkpoint_freq == 0:
                        self.save_checkpoint()

        self.save_checkpoint('Finish pretraining {:05d}.pt'.format(self.iteration))
        print('Finish pretraining trajectory')

    def add_trajectory(self, trajectory):
        self.trajectories += 1
        self.pb_trajectory_batch.trajectories.add().CopyFrom(trajectory)
        self.replay_buffer.add_trajectory(trajectory)

    def save_checkpoint(self, name='latest.pt'):
        checkpoint = {
            'iteration': self.iteration,
            'trajectories': self.trajectories,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_dir / f'{self.iteration:05d}.pt')
        torch.save(checkpoint, self.checkpoint_dir / name)

    def save_model(self):
        model = AlphaZero(**self._model_kwargs)
        model.load_state_dict(self.model.module.state_dict())
        model.eval()
        model_path = self.model_dir / f'{self.iteration:05d}.pt'

        torch.save({'model': model, 'game': self.game.observation_tensor_shape}, model_path)

        env = os.environ.copy()
        print('subprocess :', end=' ')
        process = subprocess.Popen(
            args=['python3', '-m', 'czf.utils.model_saver', '--path', model_path],
            env=env,
            universal_newlines=True
        )
        process.wait()
        print(process.pid, model_path)

        #sample_input = torch.rand(1, *self.game.observation_tensor_shape)
        #traced_model = torch.jit.trace(model, sample_input)
        # traced_model.save(str(model_path))

    async def notify_update(self):
        model_info = czf_pb2.ModelInfo(version=self.iteration)
        packet = czf_pb2.Packet(model_info=model_info)
        raw = packet.SerializeToString()
        for peer in self.peers:
            await self.send_raw(peer, raw)

    def train(self):
        trajectory_path = self.trajectory_dir / f'{self.iteration:05d}.pb.zstd'
        trajectory = self.pb_trajectory_batch.SerializeToString()
        compressed_trajectory = self.compressor.compress(trajectory)
        trajectory_path.write_bytes(compressed_trajectory)

        total_game = len(self.pb_trajectory_batch.trajectories)
        player_returns_counter = []
        for _ in range(self.game.num_players):
            player_returns_counter.append(Counter())
        game_steps = []
        for trajectory in self.pb_trajectory_batch.trajectories:
            game_steps.append(len(trajectory.states))
            for index in range(len(trajectory.statistics.rewards)):
                player_returns_counter[index].update([str(trajectory.statistics.rewards[index])])

        self.pb_trajectory_batch.Clear()

        states_to_train = int(len(self.replay_buffer) / self.replay_retention * self.replay_buffer_reuse)
        trained_states = 0

        data_loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.train()
        for observation_tensor, target_policy, target_value in data_loader:
            trained_states += len(observation_tensor)
            if (drop := trained_states - states_to_train) > 0:
                trained_states -= drop
                observation_tensor = observation_tensor[drop:]
                target_policy = target_policy[drop:]
                target_value = target_value[drop:]

            # if use transformation, sample one transformation and apply to state and policy
            if self.use_transformation:
                for i in range(len(observation_tensor)):
                    transformation_type = random.randint(0, self.game.num_transformations - 1)
                    observation = self.game.transform_observation(observation_tensor[i].flatten().numpy(), transformation_type)
                    observation_tensor[i] = torch.tensor(observation).view(self.game.observation_tensor_shape)
                    policy = self.game.transform_policy(target_policy[i].numpy(), transformation_type)
                    target_policy[i] = torch.tensor(policy)

            target_policy = target_policy.to(self.device)
            target_value = target_value.to(self.device)

            self.optimizer.zero_grad()
            policy, value = self.model.forward(observation_tensor)
            policy_loss = (-target_policy * (1e-8 + policy).log()).sum(dim=1).mean()
            value_loss = torch.nn.MSELoss()(target_value, value)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()

            # the end of current training epoch
            if drop >= 0:
                lr = next(iter(self.optimizer.param_groups))['lr']
                self.summary_writer.add_scalar('params/lr', lr, self.iteration)
                self.summary_writer.add_scalar('loss/', loss.item(), self.iteration)
                self.summary_writer.add_scalar('loss/policy', policy_loss.item(), self.iteration)
                self.summary_writer.add_scalar('loss/value', value_loss.item(), self.iteration)
                break

        self.summary_writer.add_scalar('stats/total', self.trajectories, self.iteration)
        self.summary_writer.add_scalar('stats/#games', total_game, self.iteration)
        stats_steps = {'mean': np.mean(game_steps), 'min': np.min(game_steps), 'max': np.max(game_steps), 'std': np.std(game_steps)}
        self.summary_writer.add_scalars('stats/steps', stats_steps, self.iteration)
        for i in range(len(player_returns_counter)):
            returns_frequency = {key: value/total_game for key, value in player_returns_counter[i].items()}
            self.summary_writer.add_scalars('stats/PLAYER:{}_rate'.format(i), returns_frequency, self.iteration)
            self.summary_writer.add_scalars('stats/PLAYER:{}_games'.format(i), player_returns_counter[i], self.iteration)

    async def send_raw(self, identity, raw):
        await self.socket.send_multipart([identity, raw])

    async def send_packet(self, identity, packet):
        await self.send_raw(identity, packet.SerializeToString())

    async def send_model(self, identity, model_request):
        if model_request.version == -1:
            model_request.version = self.iteration
        iteration = model_request.version
        if iteration not in self.model_response_cache:
            model_path = self.model_dir / f'{iteration:05d}.pt'
            model_response = czf_pb2.Model()
            model_response.info.CopyFrom(model_request)
            model_response.blobs.append(model_path.read_bytes())
            packet = czf_pb2.Packet(model_response=model_response)
            self.model_response_cache[iteration] = packet.SerializeToString()
        await self.send_raw(identity, self.model_response_cache[iteration])
