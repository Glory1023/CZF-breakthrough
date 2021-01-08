'''CZF Trainer'''
from collections import Counter
from io import BytesIO
import os
import subprocess
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
# import zstandard as zstd

from czf.learner.dataloader import RolloutBatch
# from czf.learner.distributed import DistributedDataParallelWrapper
from czf.learner.data_parallel import DataParallelWrapper
from czf.learner.nn import MuZero, MuZeroAtari


class Trainer:
    '''Trainer'''
    def __init__(self,
                 config,
                 checkpoint_path,
                 model_path,
                 log_path,
                 num_proc,
                 model_name,
                 restore,
                 rank=0):
        self._device = 'cuda'
        self.model_name, self.iteration = model_name, 0
        self._ckpt_dir = checkpoint_path / self.model_name
        self._model_dir = model_path / self.model_name
        for path in (self._ckpt_dir, self._model_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._num_proc = num_proc
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
        self._r_heads = 2 * r_heads + 1 if has_transform else r_heads
        self._v_heads = 2 * v_heads + 1 if has_transform else v_heads
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
        self._replay_buffer_reuse = config['learner']['replay_buffer_reuse']
        self._replay_retention = config['learner'][
            'replay_buffer_size'] / config['learner']['frequency']
        self._rollout_steps = config['learner']['rollout_steps']
        self.frequency = config['learner']['frequency']
        self._batch_size = config['learner']['batch_size']
        self._r_loss = model_config['r_loss']
        self._v_loss = model_config['v_loss']
        # model
        # torch.cuda.set_device(rank)
        # torch.distributed.init_process_group('nccl', init_method='env://')
        Model = MuZeroAtari if (self._model_cls == 'MuZeroAtari') else MuZero
        self._model = Model(
            **self._model_kwargs,
            is_train=True,
        ).to(self._device)
        # self._model = DistributedDataParallelWrapper(self._model,
        #                                   device_ids=[rank],
        #                                   output_device=rank)
        # restore the latest checkpoint
        if restore:
            print('Restore from', restore)
            with open(restore, 'rb') as model_blob:
                buffer = model_blob.read()
                # dctx = zstd.ZstdDecompressor()
                # buffer = dctx.decompress(buffer)
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
        self._summary_writer = SummaryWriter(log_dir=log_path,
                                             purge_step=self.iteration)
        # note: PyTorch supports the `forward` method currently
        # so, we can only trace the prediction model now.
        # input_obs = torch.rand(1, *observation_shape).to(self._device)
        input_state = torch.rand(1, *state_shape).to(self._device)
        # input_action = torch.rand(1, 1).to(self._device)
        self._summary_writer.add_graph(self._model.module, (input_state, ))

    def log_statistics(self, replay_buffer):
        '''log statistics for recent trajectories'''
        statistics = replay_buffer.get_statistics()
        replay_buffer.reset_statistics()
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('game/num_games', statistics.num_games, step)
        game_steps = statistics.game_steps
        if game_steps:
            writer.add_scalar('game/num_states', sum(game_steps), step)
            writer.add_scalars(
                'game/steps', {
                    'mean': np.mean(game_steps),
                    'min': np.min(game_steps),
                    'max': np.max(game_steps),
                    'std': np.std(game_steps),
                }, step)
        if self._num_player == 1:
            score = statistics.player_returns[0]
            if score:
                writer.add_scalars(
                    'game/score', {
                        'mean': np.mean(score),
                        'min': np.min(score),
                        'max': np.max(score),
                        'std': np.std(score),
                    }, step)
        else:
            for p, player_returns in enumerate(statistics.player_returns):
                player_returns = Counter(player_returns)
                returns_rates = {
                    str(key): value / statistics.num_games
                    for key, value in player_returns.items()
                }
                writer.add_scalars(f'game/player{p}_rate', returns_rates, step)

    def train(self, replay_buffer):
        '''distributed training wrapper'''
        # with self._model.no_sync():
        self._train(replay_buffer)
        self.iteration += 1

    def _train(self, replay_buffer):
        '''optimize the model and increment model version'''
        p_criterion = lambda target, prob: ((-target *
                                             (1e-8 + prob).log()).sum(dim=1))
        if self._v_loss == 'mse':
            v_criterion = lambda target, pred: (target - pred).square(
            ).squeeze()
        else:  # == 'cross_entropy'
            v_criterion = lambda target, prob: (
                (-target * (1e-8 + prob).log()).sum(dim=1))
        if self._r_loss == 'mse':
            r_criterion = lambda target, pred: torch.nn.MSELoss(
                reduction='none')(target, pred).squeeze()
        else:  # == 'cross_entropy'
            r_criterion = lambda target, prob: (
                (-target * (1e-8 + prob).log()).sum(dim=1))
        scale_gradient = lambda tensor, scale: (tensor * scale + tensor.detach(
        ) * (1 - scale))
        states_to_train = int(
            len(replay_buffer) / self._replay_retention *
            self._replay_buffer_reuse)
        sampler = WeightedRandomSampler(replay_buffer.get_weights(),
                                        states_to_train)
        dataloader = DataLoader(
            dataset=replay_buffer,
            batch_size=self._batch_size,
            sampler=sampler,
            # shuffle=True, # shuffle and sampler are disjoint
            collate_fn=RolloutBatch,
            pin_memory=True,
            prefetch_factor=4 if self._num_proc > 0 else 0,
            num_workers=self._num_proc,
        )
        to_tensor = lambda x, dtype=np.float32: torch.as_tensor(
            np.frombuffer(x, dtype=dtype), device=self._device)
        shape = (
            (-1, self._v_heads),
            (-1, ),
            (-1, self._action_dim),
            (-1, ),
            (-1, self._r_heads),
        )
        self._model.train()
        num_trained_states = 0
        replay_buffer.copy_weights()
        print(f'Weights mean: {replay_buffer._buffer._weights_mean:.5f}')
        if self._model_cls == 'MuZeroAtari':
            value_transform = lambda x: MuZeroAtari.to_scalar(
                self._model.v_supp, x)
            reward_transform = lambda x: MuZeroAtari.to_scalar(
                self._model.r_supp, x)
        else:
            value_transform = lambda x: x
            reward_transform = lambda x: x
        for rollout in dataloader:
            # tensor
            weight = to_tensor(rollout.weight)
            observation = to_tensor(rollout.observation).view(
                -1, *self._observation_shape)
            scale = to_tensor(rollout.scale)
            transition = [
                to_tensor(t).view(shape[i % 5])
                for i, t in enumerate(rollout.transition)
            ]
            num_states = len(observation)
            num_trained_states += num_states
            # priority
            nstep_v = torch.zeros((num_states, 1), device=self._device)
            nstep_v_sum = torch.zeros((num_states, 1), device=self._device)
            ksteps = torch.zeros((num_states, 1), device=self._device)
            rollout_index = torch.arange(num_states, device=self._device)
            target_v_info = value_transform(transition[0])
            scalar_v = value_transform(transition[0])
            # forward
            state = self._model.parallel_forward_representation(observation)
            total_batch, loss, p_loss, v_loss, r_loss = 0, 0, 0, 0, 0
            for t, i in enumerate(range(0, len(transition), 5)):
                target_value, mask, *next_transition = transition[i:i + 5]
                if next_transition:
                    target_policy, action, target_reward = next_transition
                mask = mask.nonzero(as_tuple=True)
                total_batch += len(state)
                policy, value = self._model.parallel_forward(state)
                # predict nstep
                nstep_v[rollout_index] += value_transform(
                    value.detach()) * (self._gamma**t)
                nstep_v_sum[rollout_index] += nstep_v[
                    rollout_index] if self._num_player == 1 else value_transform(
                        value.detach()) * (self._gamma**t) * ((-1)**t)
                ksteps[rollout_index] += 1.
                target_v_info[rollout_index] += value_transform(
                    target_value
                ) if self._num_player == 1 else value_transform(
                    target_value) * ((-1)**t)
                if t == 0 and self.use_prioritize:
                    priority = torch.abs(
                        value_transform(value.detach()) -
                        value_transform(target_value)).squeeze(
                            dim=-1).tolist()
                    replay_buffer.update_weights(rollout.index, priority)
                if next_transition:
                    policy = policy[mask]
                    state = state[mask]
                    state, reward = self._model.parallel_forward_dynamics(
                        state, action)
                    state = scale_gradient(state, 0.5)
                    # predict nstep
                    rollout_index = rollout_index[mask]
                    nstep_v[rollout_index] += reward_transform(
                        reward.detach()) * (self._gamma**t) - value_transform(
                                value[mask].detach()) * (self._gamma**t)
                    if t == 0:
                    print('avg target policy',
                              torch.mean(target_policy, dim=0),
                              'avg pred policy', torch.mean(policy, dim=0))
                # loss
                v_loss_i = weight * v_criterion(target_value, value)
                if next_transition:
                    masked_weight = weight[mask]
                    p_loss_i = masked_weight * p_criterion(
                        target_policy, policy)
                    r_loss_i = masked_weight * r_criterion(
                        target_reward, reward)
                    weight = masked_weight
                # scale gradient
                if i > 0:
                    v_loss_i = scale_gradient(v_loss_i, scale)
                if next_transition:
                    scale = scale[mask]
                    if i > 0:
                        p_loss_i = scale_gradient(p_loss_i, scale)
                    r_loss_i = scale_gradient(r_loss_i, scale)
                # total loss
                v_loss_i = v_loss_i.sum()
                loss += v_loss_i
                if next_transition:
                    p_loss_i = p_loss_i.sum()
                    r_loss_i = r_loss_i.sum()
                    loss += p_loss_i + r_loss_i
                v_loss += v_loss_i.item()
                if next_transition:
                    p_loss += p_loss_i.item()
                    r_loss += r_loss_i.item()
            loss /= total_batch
            p_loss /= total_batch
            v_loss /= total_batch
            r_loss /= total_batch
            nstep_v_sum /= ksteps
            target_v_info /= ksteps
            print(
                'p_loss: {:.3f}, v_loss: {:.3f}, target_v: {:.3f} \u00b1 {:.3f}, rollout_v: {:.3f} \u00b1 {:.3f}, priority ori: {:.3f}'
                .format(
                    p_loss, v_loss, torch.mean(target_v_info.detach()),
                    torch.std(target_v_info.detach()),
                    torch.mean(nstep_v_sum.detach()),
                    torch.std(nstep_v_sum.detach()),
                    torch.mean(replay_buffer._buffer._weights_mean /
                               to_tensor(rollout.weight))))
            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 40)
            self._optimizer.step()
            loss = loss.item()
            del weight, observation, scale, transition
            del rollout, target_value, mask, next_transition
            del state, action, policy, value, reward
            del scalar_v, nstep_v, nstep_v_sum, ksteps, rollout_index, target_v_info
            # the end of current training
            if num_trained_states >= states_to_train:
                # TODO: average loss of each minibatch
                lr = next(iter(self._optimizer.param_groups))['lr']
                writer, step = self._summary_writer, self.iteration
                writer.add_scalar('params/lr', lr, step)
                writer.add_scalar('loss/', loss, step)
                writer.add_scalar('loss/policy', p_loss, step)
                writer.add_scalar('loss/value', v_loss, step)
                writer.add_scalar('loss/reward', r_loss, step)
                break
        replay_buffer.write_back_weights()

    def save_model(self, checkpoint=False):
        '''save model to file'''
        buffer = BytesIO()
        torch.save(
            {
                'name': self.model_name,
                'iteration': self.iteration,
                'model': self._model.module,
                'optimizer': self._optimizer.state_dict(),
                'observation_shape': self._observation_shape,
                'state_shape': self._state_shape,
            }, buffer)
        buffer.seek(0)
        buffer = buffer.read()
        # buffer = self._ckpt_compressor.compress(buffer)
        ckpt_path = self._ckpt_dir / f'{self.iteration:05d}.pt.zst'
        ckpt_path.write_bytes(buffer)
        # frozen model
        args = [
            sys.executable,
            '-m',
            'czf.utils.model_saver',
            '--checkpoint',
            str(ckpt_path),
            '--model-dir',
            str(self._model_dir),
        ]
        if not checkpoint:
            args.append('--rm')
        # subprocess.run(args, check=True, env=os.environ.copy())
        # if checkpoint:
        if True:
            # update the latest checkpoint
            latest_ckpt = self._ckpt_dir / 'latest.pt.zst'
            temp_ckpt = self._ckpt_dir / 'latest-temp.pt.zst'
            os.symlink(ckpt_path, temp_ckpt)
            os.replace(temp_ckpt, latest_ckpt)
