'''CZF AlphaZero Trainer'''
from collections import Counter
from io import BytesIO
import os
import subprocess
import sys

import numpy as np
import psutil
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter

from czf.env import czf_env
from czf.learner.nn import AlphaZero
from czf.learner.trainer.trainer import Trainer


class AlphaZeroTrainer(Trainer):
    '''AlphaZero Trainer'''
    def __init__(self, config, checkpoint_path, model_path, log_path, model_name, restore, gpus):
        self._device = 'cuda:' + str(gpus[0]) if len(gpus) > 0 else 'cuda'
        self.model_name, self.iteration = model_name, 0
        self._ckpt_dir = checkpoint_path / self.model_name
        self._model_dir = model_path / self.model_name
        for path in (self._ckpt_dir, self._model_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_freq = config['learner']['checkpoint_freq']
        self._game = czf_env.load_game(config['game']['name'])
        model_config = config['model']
        self._model_kwargs = dict(
            observation_tensor_shape=self._game.observation_tensor_shape,
            action_dim=self._game.num_distinct_actions,
            channels=model_config['channels'],
            blocks=model_config['blocks'],
            v_heads=self._game.num_players,
            backbone=model_config.get('backbone', 'ResNet'),
        )
        self._model = AlphaZero(**self._model_kwargs)
        device_ids = gpus if len(gpus) > 0 else None
        self._model = torch.nn.DataParallel(
            self._model,
            device_ids=device_ids,
        )
        self._model.to(self._device)
        # optimizer
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=config['learner']['optimizer']['learning_rate'],
            momentum=config['learner']['optimizer']['momentum'],
            weight_decay=config['learner']['optimizer']['weight_decay'],
            nesterov=config['learner']['optimizer']['nesterov'],
        )
        # learning rate scheduler
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer,
            milestones=config['learner']['lr_scheduler']['milestones'],
            gamma=config['learner']['lr_scheduler']['gamma'],
        )
        # restore the latest checkpoint
        # latest_checkpoint = torch.load(self._ckpt_dir / args.checkpoint) if args.restore else None
        # if latest_checkpoint:
        #     self.iteration = latest_checkpoint['iteration']
        #     self._model.load_state_dict(latest_checkpoint['model'])
        #     self._optimizer.load_state_dict(latest_checkpoint['optimizer'])
        #     self._scheduler.load_state_dict(latest_checkpoint['scheduler'])
        # self.restore_replay_buffer()

        # tensorboard log
        self._num_player = config['game']['num_player']
        self._summary_writer = SummaryWriter(
            log_dir=log_path,
            purge_step=self.iteration,
        )

        # if args.pretrain_trajectory_dir:
        #     print('pretrain trajectory directory:', args.pretrain_trajectory_dir)
        #     self.pretrain_trajectory(args.pretrain_trajectory_dir)

    def train(self, dataloader, replay_buffer):
        '''distributed training wrapper'''
        self._train(dataloader, replay_buffer)
        self.iteration += 1

    def _train(self, dataloader, replay_buffer):
        self._model.train()
        to_tensor = lambda x, dtype=np.float32: torch.as_tensor(
            np.array(np.frombuffer(x, dtype=dtype)), device=self._device)
        # for observation_tensor, target_policy, target_value in dataloader:
        for transition in dataloader:
            observation_tensor = to_tensor(transition.observation).view(
                -1, *self._game.observation_tensor_shape)
            target_policy = to_tensor(transition.policy).view(-1, self._game.num_distinct_actions)
            target_value = to_tensor(transition.value).view(-1, self._game.num_players)

            self._optimizer.zero_grad()
            policy, value = self._model.forward(observation_tensor)
            policy_loss = (-target_policy * (1e-8 + policy).log()).sum(dim=1).mean()
            value_loss = torch.nn.MSELoss()(target_value, value)
            loss = policy_loss + value_loss
            loss.backward()
            self._optimizer.step()

        # the end of current training epoch
        lr = next(iter(self._optimizer.param_groups))['lr']
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('params/lr', lr, step)
        writer.add_scalar('loss/', loss.item(), step)
        writer.add_scalar('loss/policy', policy_loss.item(), step)
        writer.add_scalar('loss/value', value_loss.item(), step)

    def log_statistics(self, replay_buffer):
        '''log statistics for recent trajectories'''
        get_logging_dict = lambda x: dict(
            mean=np.mean(x), min=np.min(x), max=np.max(x), std=np.std(x))
        statistics = replay_buffer.get_statistics()
        replay_buffer.reset_statistics()
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('game/num_games', statistics.num_games, step)
        writer.add_scalar('game/num_states', statistics.num_states, step)
        game_steps = statistics.game_steps
        if game_steps:
            writer.add_scalars('game/steps', get_logging_dict(game_steps), step)
        if self._num_player == 1:
            score = statistics.player_returns[0]
            writer.add_scalars('game/score', get_logging_dict(score), step)
        else:
            for p, player_returns in enumerate(statistics.player_returns):
                player_returns = Counter(player_returns)
                returns_rates = {
                    str(key): value / statistics.num_games
                    for key, value in player_returns.items()
                }
                writer.add_scalars(f'game/player{p}_rate', returns_rates, step)
        # memory usage (check memory leak in pytorch)
        process = psutil.Process()
        process_memory = process.memory_info()
        for name in process_memory._fields:
            value = getattr(process_memory, name)
            writer.add_scalar('memory/{}'.format(name.capitalize()), value, self.iteration)

    def save_model(self):
        '''save model to file'''
        # save the checkpoint of current iteration
        buffer_ = BytesIO()
        torch.save(
            {
                'name': self.model_name,
                'iteration': self.iteration,
                'observation_shape': self._game.observation_tensor_shape,
                'model': self._model.module,
                'optimizer': self._optimizer.state_dict(),
                'scheduler': self._scheduler.state_dict()
            }, buffer_)
        buffer_.seek(0)
        buffer_ = buffer_.read()
        ckpt_path = self._ckpt_dir / f'{self.iteration:05d}.pt'
        ckpt_path.write_bytes(buffer_)
        # frozen model
        args = [
            sys.executable,
            '-m',
            'czf.utils.model_saver',
            '--checkpoint-path',
            str(ckpt_path),
            '--model-dir',
            str(self._model_dir),
            '--algorithm',
            'AlphaZero',
            '--device',
            self._device,
        ]
        subprocess.run(args, check=True, env=os.environ.copy())

        prev_iteration = self.iteration - 1
        remove_prev_ckpt = (prev_iteration >= 0) and (prev_iteration % self._checkpoint_freq != 0)
        if remove_prev_ckpt:
            # remove previous checkpoint
            prev_ckpt_path = self._ckpt_dir / f'{prev_iteration:05d}.pt'
            if os.path.exists(str(prev_ckpt_path)):
                os.remove(str(prev_ckpt_path))
            # remove previous model
            prev_model_path = self._model_dir / f'{prev_iteration:05d}.pt'
            if os.path.exists(str(prev_model_path)):
                os.remove(str(prev_model_path))

        keep_current_ckpt = (self.iteration % self._checkpoint_freq == 0)
        if keep_current_ckpt:
            # update the link to latest checkpoint
            latest_ckpt = self._ckpt_dir / 'latest.pt'
            temp_ckpt = self._ckpt_dir / 'latest-temp.pt'
            os.symlink(ckpt_path, temp_ckpt)
            os.replace(temp_ckpt, latest_ckpt)
            # update the link to latest model
            latest_model = self._model_dir / 'latest.pt'
            temp_model = self._model_dir / 'latest-temp.pt'
            model_path = self._model_dir / f'{self.iteration:05d}.pt'
            os.symlink(model_path, temp_model)
            os.replace(temp_model, latest_model)
