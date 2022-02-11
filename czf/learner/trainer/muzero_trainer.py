'''CZF MuZero Trainer'''
from collections import Counter
from io import BytesIO
import os
import subprocess
import sys

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

from czf.learner.data_parallel import DataParallelWrapper
from czf.learner.nn import MuZero, MuZeroAtari
from czf.learner.trainer.trainer import Trainer


class MuZeroTrainer(Trainer):
    '''MuZero Trainer'''
    def __init__(self, config, checkpoint_path, model_path, log_path, model_name, restore):
        self._device = 'cuda'
        self.model_name, self.iteration = model_name, 0
        self._ckpt_dir = checkpoint_path / self.model_name
        self._model_dir = model_path / self.model_name
        for path in (self._ckpt_dir, self._model_dir):
            path.mkdir(parents=True, exist_ok=True)
        # game
        observation_config = config['game']['observation']
        frame_stack = observation_config['frame_stack']
        channel = observation_config['channel']
        spatial_shape = observation_config['spatial_shape']
        if frame_stack > 0:
            observation_shape = [frame_stack * (channel + 1), *spatial_shape]
        else:
            observation_shape = [channel, *spatial_shape]
        self._gamma = config['mcts']['discount_factor']
        # model kwargs
        self._action_dim = config['game']['actions']
        model_config = config['model']
        h_channels = model_config['h_channels']
        state_shape = [h_channels, *config['game']['state_spatial_shape']]
        self._model_cls = config['model']['name']
        has_transform = 'transform' in config['learner']
        r_heads = model_config.get('r_heads', 1)
        v_heads = model_config['v_heads']
        self._r_heads = r_heads[1] - r_heads[0] + 1 if has_transform else r_heads
        self._v_heads = v_heads[1] - v_heads[0] + 1 if has_transform else v_heads
        self._has_transform = None
        self._model_kwargs = dict(
            observation_shape=observation_shape,
            state_shape=state_shape,
            action_dim=self._action_dim,
            h_blocks=model_config['h_blocks'],
            h_channels=h_channels,
            g_blocks=model_config['g_blocks'],
            r_heads=r_heads,
            f_blocks=model_config['f_blocks'],
            f_channels=model_config['f_channels'],
            v_heads=v_heads,
        )
        # config
        self._observation_shape = observation_shape
        self._state_shape = state_shape
        self._use_prioritize = config['learner']['prioritized']
        self._kstep = config['learner']['rollout_steps']
        self._batch_size = config['learner']['batch_size']
        self._checkpoint_freq = config['learner']['checkpoint_freq']
        self._r_loss = model_config['r_loss']
        self._v_loss = model_config['v_loss']
        # model
        Model = MuZeroAtari if (self._model_cls == 'MuZeroAtari') else MuZero
        self._model = Model(
            **self._model_kwargs,
            is_train=True,
        ).to(self._device)
        # loss function
        self._p_loss_func = lambda target, prob: ((-target * (1e-8 + prob).log()).sum(dim=1))
        if self._v_loss == 'mse':
            self._v_loss_func = lambda target, pred: (target - pred).square().squeeze()
        else:  # == 'cross_entropy'
            self._v_loss_func = lambda target, prob: ((-target * (1e-8 + prob).log()).sum(dim=1))
        if self._r_loss == 'mse':
            self._r_loss_func = lambda target, pred: torch.nn.MSELoss(reduction='none')(
                target, pred).squeeze()
        else:  # == 'cross_entropy'
            self._r_loss_func = lambda target, prob: ((-target * (1e-8 + prob).log()).sum(dim=1))
        # transform function
        if self._model_cls == 'MuZeroAtari':
            self._value_transform = lambda x: MuZeroAtari.to_scalar(self._model.v_supp, x)
            self._reward_transform = lambda x: MuZeroAtari.to_scalar(self._model.r_supp, x)
        else:
            self._value_transform = lambda x: x
            self._reward_transform = lambda x: x

        # restore the latest checkpoint
        if restore:
            print('Restore from', restore)
            with open(restore, 'rb') as model_blob:
                buffer = model_blob.read()
            state_dict = torch.load(BytesIO(buffer))
            self.iteration = state_dict['iteration']
            self._model = state_dict['model']
            # self._optimizer.load_state_dict(state_dict['optimizer'])

        self._model = DataParallelWrapper(self._model)
        # optimizer
        torch.backends.cudnn.benchmark = True
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=config['learner']['optimizer']['learning_rate'],
            momentum=config['learner']['optimizer']['momentum'],
            weight_decay=config['learner']['optimizer']['weight_decay'],
            nesterov=config['learner']['optimizer']['nesterov'],
        )
        # tensorboard log
        self._num_player = config['game']['num_player']
        self._summary_writer = SummaryWriter(
            log_dir=log_path,
            purge_step=self.iteration,
        )
        # note: PyTorch supports the `forward` method currently
        # so, we can only trace the prediction model now.
        input_state = torch.rand(1, *state_shape).to(self._device)
        self._summary_writer.add_graph(self._model.module, (input_state, ))

    def train(self, dataloader, replay_buffer):
        '''distributed training wrapper'''
        self._train(dataloader, replay_buffer)
        self.iteration += 1

    def _train(self, dataloader, replay_buffer):
        '''optimize the model and increment model version'''
        scale_gradient = lambda tensor, scale: (tensor * scale + tensor.detach() * (1 - scale))
        to_tensor = lambda x, dtype=np.float32: torch.as_tensor(
            np.array(np.frombuffer(x, dtype=dtype)), device=self._device)
        # shape for one transition
        shape = (
            (-1, self._v_heads),
            (-1, ),
            (-1, self._action_dim),
            (-1, ),
            (-1, self._r_heads),
        )

        self._model.train()
        replay_buffer.copy_weights()
        print(f'>> Priority Mean: {replay_buffer.get_mean_weight():.5f}')

        # logging
        value_target, value_rollout = [], []
        policy_target, policy_rollout = [], []
        reward_sum_target, reward_sum_rollout = [], []
        priority_sampled = []

        for rollout in dataloader:
            # tensor
            weight = to_tensor(rollout.weight)
            observation = to_tensor(rollout.observation).view(-1, *self._observation_shape)
            scale = to_tensor(rollout.scale)
            transition = [
                to_tensor(t).view(shape[i % len(shape)]) for i, t in enumerate(rollout.transition)
            ]

            total_batch, loss, p_loss, v_loss, r_loss = 0, 0, 0, 0, 0
            # logging info
            num_states = len(observation)
            ksteps = torch.zeros((num_states, 1), device=self._device)
            rollout_index = torch.arange(num_states, device=self._device)
            target_v = torch.zeros((num_states, 1), device=self._device)
            rollout_v = torch.zeros((num_states, 1), device=self._device)
            target_r_sum = torch.zeros((num_states, 1), device=self._device)
            rollout_r_sum = torch.zeros((num_states, 1), device=self._device)

            state = self._model.parallel_forward_representation(observation)
            # rollout k steps
            for t, i in enumerate(range(0, len(transition), 5)):
                is_last_step = (t == self._kstep)
                target_value, mask, *next_transition = transition[i:i + 5]
                if next_transition:
                    target_policy, action, target_reward = next_transition
                mask = mask.nonzero(as_tuple=True)
                total_batch += len(state)
                policy, value = self._model.parallel_forward(state)
                ksteps[rollout_index] += 1.

                # logging info
                if self._num_player == 1:
                    target_v[rollout_index] += self._value_transform(target_value)
                else:
                    target_v[rollout_index] += self._value_transform(target_value) * ((-1)**t)
                if self._num_player == 1:
                    rollout_v[rollout_index] += self._value_transform(value.detach())
                else:
                    rollout_v[rollout_index] += self._value_transform(value.detach()) * ((-1)**t)

                # priority
                if t == 0 and self._use_prioritize:
                    priority = torch.abs(
                        self._value_transform(value.detach()) -
                        self._value_transform(target_value)).squeeze(dim=-1).tolist()
                    replay_buffer.update_weights(rollout.index, priority)

                if next_transition:
                    policy = policy[mask]
                    state = state[mask]
                    state, reward = self._model.parallel_forward_dynamics(state, action)
                    state = scale_gradient(state, 0.5)
                    rollout_index = rollout_index[mask]
                    # logging info
                    if not is_last_step:
                        target_r_sum[rollout_index] += self._reward_transform(target_reward)
                        rollout_r_sum[rollout_index] += self._reward_transform(reward.detach())
                    policy_target.extend(target_policy.tolist())
                    policy_rollout.extend(policy.detach().tolist())
                    if t == 0 and self._num_player == 1:
                        print('>>> avg target_p:',
                              [round(p, 3)
                               for p in torch.mean(target_policy, dim=0).tolist()], 'avg pred_p:',
                              [round(p, 3) for p in torch.mean(policy, dim=0).tolist()])

                # loss
                v_loss_i = weight * self._v_loss_func(target_value, value)
                if next_transition:
                    masked_weight = weight[mask]
                    p_loss_i = masked_weight * self._p_loss_func(target_policy, policy)
                    if not is_last_step:
                        r_loss_i = masked_weight * self._r_loss_func(target_reward, reward)
                    weight = masked_weight
                # scale gradient
                if i > 0:
                    v_loss_i = scale_gradient(v_loss_i, scale)
                if next_transition:
                    scale = scale[mask]
                    if i > 0:
                        p_loss_i = scale_gradient(p_loss_i, scale)
                    if not is_last_step:
                        r_loss_i = scale_gradient(r_loss_i, scale)
                # total loss
                v_loss_i = v_loss_i.sum()
                loss += v_loss_i
                if next_transition:
                    p_loss_i = p_loss_i.sum()
                    loss += p_loss_i
                    if not is_last_step:
                        r_loss_i = r_loss_i.sum()
                        loss += r_loss_i
                v_loss += v_loss_i.item()
                if next_transition:
                    p_loss += p_loss_i.item()
                    if not is_last_step:
                        r_loss += r_loss_i.item()

            loss /= total_batch
            p_loss /= total_batch
            v_loss /= total_batch
            r_loss /= total_batch
            target_v /= ksteps
            rollout_v /= ksteps
            # logging
            value_target.extend(target_v.tolist())
            value_rollout.extend(rollout_v.tolist())
            reward_sum_rollout.extend(rollout_r_sum.tolist())
            reward_sum_target.extend(target_r_sum.tolist())
            priority_sampled.extend(
                (replay_buffer.get_mean_weight() / to_tensor(rollout.weight)).tolist())
            print(
                '... p_loss: {:.3f}, v_loss: {:.3f}, r_loss: {:.3f}, priority: {:.3f} \u00b1 {:.3f}'
                .format(p_loss, v_loss, r_loss,
                        torch.mean(replay_buffer.get_mean_weight() / to_tensor(rollout.weight)),
                        torch.std(replay_buffer.get_mean_weight() / to_tensor(rollout.weight))))
            print(
                '... target_v: {:.3f} \u00b1 {:.3f}, rollout_v: {:.3f} \u00b1 {:.3f},'
                ' target_r_sum: {:.3f} \u00b1 {:.3f}, rollout_r_sum: {:.3f} \u00b1 {:.3f}'.format(
                    torch.mean(target_v.detach()),
                    torch.std(target_v.detach()),
                    torch.mean(rollout_v.detach()),
                    torch.std(rollout_v.detach()),
                    torch.mean(target_r_sum),
                    torch.std(target_r_sum),
                    torch.mean(rollout_r_sum),
                    torch.std(rollout_r_sum),
                ))
            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 40)
            self._optimizer.step()
            loss = loss.item()
            del weight, observation, scale, transition
            del rollout, target_value, mask, next_transition
            del state, action, policy, value, reward
            del ksteps, rollout_index, target_v, rollout_v, target_r_sum, rollout_r_sum
        # the end of current training
        # logging: loss
        # TODO: average loss of each minibatch?
        lr = next(iter(self._optimizer.param_groups))['lr']
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('params/lr', lr, step)
        writer.add_scalar('loss/', loss, step)
        writer.add_scalar('loss/policy', p_loss, step)
        writer.add_scalar('loss/value', v_loss, step)
        writer.add_scalar('loss/reward', r_loss, step)
        # logging: train
        get_logging_dict = lambda x: dict(
            mean=np.mean(x), min=np.min(x), max=np.max(x), std=np.std(x))
        writer.add_scalars('train/policy_target',
                           {f'action {i}': np.mean(p)
                            for i, p in enumerate(zip(*policy_target))}, step)
        writer.add_scalars('train/policy_rollout',
                           {f'action {i}': np.mean(p)
                            for i, p in enumerate(zip(*policy_rollout))}, step)
        writer.add_scalars('train/value_target', get_logging_dict(value_target), step)
        writer.add_scalars('train/value_rollout', get_logging_dict(value_rollout), step)
        writer.add_scalars('train/reward_target', get_logging_dict(reward_sum_target), step)
        writer.add_scalars('train/reward_rollout', get_logging_dict(reward_sum_rollout), step)
        writer.add_scalars('train/priority', get_logging_dict(priority_sampled), step)
        # write back priorities
        replay_buffer.write_back_weights()

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
            if score:
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
                'model': self._model.module,
                'optimizer': self._optimizer.state_dict(),
                'observation_shape': self._observation_shape,
                'state_shape': self._state_shape,
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
            'MuZero',
        ]
        subprocess.run(args, check=True, env=os.environ.copy())

        prev_iteration = self.iteration - 1
        remove_prev_ckpt = (prev_iteration >= 0) and (prev_iteration % self._checkpoint_freq != 0)
        if remove_prev_ckpt:
            # remove previous checkpoint
            prev_ckpt_path = self._ckpt_dir / f'{prev_iteration:05d}.pt'
            os.remove(str(prev_ckpt_path))
            # remove previous model
            prev_model_path = self._model_dir / f'{prev_iteration:05d}.pt'
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
