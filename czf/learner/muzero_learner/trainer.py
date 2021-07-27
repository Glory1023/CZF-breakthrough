'''CZF Trainer'''
from collections import Counter
from datetime import datetime
from io import BytesIO
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
import subprocess
import sys
from queue import Empty
import time
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# import zstandard as zstd

from czf.learner.muzero_learner.replay_buffer import ReplayBuffer, RolloutBatch
from czf.learner.muzero_learner.data_parallel import DataParallelWrapper
from czf.learner.nn import MuZero, MuZeroAtari


def run_sampler(index_queue, sample_queue, replay_buffer, prefetch_factor):
    '''run sampler'''
    while True:
        index = [index_queue.get()]
        try:
            for _ in range(prefetch_factor):
                index.append(index_queue.get(block=False))
        except Empty:
            pass
        finally:
            for i in index:
                sample_queue.put(replay_buffer[i])


class MyDataLoader:
    '''DataLoader'''
    def __init__(
        self,
        index_queue,
        sample_queue,
        batch_size,
        collate_fn,
    ):
        self._index_queue = index_queue
        self._sample_queue = sample_queue
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._num_sample = 0

    def put(self, sampler):
        '''Put all samples from `sampler` into `index_queue`'''
        for index in sampler:
            self._num_sample += 1
            self._index_queue.put(index)

    def __iter__(self):
        while self._num_sample > self._batch_size:
            data = self._collate_fn(
                [self._sample_queue.get() for _ in range(self._batch_size)])
            self._num_sample -= self._batch_size
            yield data
        data = self._collate_fn(
            [self._sample_queue.get() for _ in range(self._num_sample)])
        self._num_sample = 0
        yield data


def run_trainer(args, config, path, trajectory_queue, notify_model_queue):
    '''run :class:`Trainer`'''
    storage_path, checkpoint_path, model_path, log_path, trajectory_path = path
    # replay buffer
    BaseManager.register('ReplayBuffer',
                         ReplayBuffer,
                         exposed=[
                             '__len__',
                             '__getitem__',
                             'get_mean_weight',
                             'get_weights',
                             'update_weights',
                             'copy_weights',
                             'write_back_weights',
                             'get_num_to_add',
                             'get_states_to_train',
                             'add_trajectory',
                             'get_statistics',
                             'reset_statistics',
                             'save_trajectory',
                         ])
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(
        num_player=config['game']['num_player'],
        observation_config=config['game']['observation'],
        kstep=config['learner']['rollout_steps'],
        capacity=config['learner']['replay_buffer_size'],
        states_to_train=config['learner'].get('states_to_train', None),
        sequences_to_train=config['learner'].get('sequences_to_train', None),
        sample_ratio=config['learner'].get('sample_ratio', None),
        sample_states=config['learner'].get('sample_states', None),
    )
    # restore the replay buffer
    if args.restore_buffer_dir:
        pass  # TODO: restore the replay buffer
    # sampler
    index_queue = mp.Queue()
    sample_queue = mp.Queue()
    prefetch_factor = 4
    samplers = [
        mp.Process(target=run_sampler,
                   args=(
                       index_queue,
                       sample_queue,
                       replay_buffer,
                       prefetch_factor,
                   )) for _ in range(args.num_proc)
    ]
    for sampler in samplers:
        sampler.start()
    # trainer
    checkpoint_freq = checkpoint_freq = config['learner']['checkpoint_freq']
    trainer = Trainer(config, checkpoint_path, model_path, log_path,
                      args.model_name, args.restore_checkpoint_path,
                      config['learner']['prioritized'])
    trainer.save_model()
    print('Storage path:', storage_path)
    # pretrain the trajectory
    if args.pretrain_trajectory_dir:
        pass  # TODO: pretrain the trajectory
    # dataloader
    dataloader = MyDataLoader(
        index_queue,
        sample_queue,
        batch_size=config['learner']['batch_size'],
        collate_fn=RolloutBatch,
    )
    pbar = tqdm(total=replay_buffer.get_num_to_add(), desc='Collect Trajs')
    # training loop
    while True:
        trajectories = trajectory_queue.get_all()
        progress = 0
        for trajectory in trajectories:
            progress += replay_buffer.add_trajectory(trajectory)
        pbar.update(min(progress, pbar.total - pbar.n))
        states_to_train = replay_buffer.get_states_to_train()
        if states_to_train > 0:
            pbar.close()
            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] >> '
                  'Start optimization')
            print('>> States to train:', states_to_train)
            start = time.time()
            trainer.log_statistics(replay_buffer)
            replay_buffer.save_trajectory(trajectory_path, trainer.iteration)
            sampler = WeightedRandomSampler(replay_buffer.get_weights(),
                                            states_to_train)
            dataloader.put(sampler)
            trainer.train(dataloader, replay_buffer)
            save_ckpt = (trainer.iteration % checkpoint_freq == 0)
            trainer.save_model(save_ckpt)
            notify_model_queue.put((trainer.model_name, trainer.iteration))
            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] >> '
                  f'Finish optimization with time {time.time() - start:.3f}')
            pbar = tqdm(total=replay_buffer.get_num_to_add(),
                        desc='Collect Trajs')


class TrainerRunner:
    '''Manage :class:`Trainer` to run in another process.'''
    def __init__(self, args, config, path, trajectory_queue):
        self._notify_model_queue = mp.Queue()
        self._trainer = mp.Process(target=run_trainer,
                                   args=(args, config, path, trajectory_queue,
                                         self._notify_model_queue))
        self._trainer.start()

    def get_notify(self):
        '''get notify for a new model'''
        return self._notify_model_queue.get()


class Trainer:
    '''Trainer'''
    def __init__(self,
                 config,
                 checkpoint_path,
                 model_path,
                 log_path,
                 model_name,
                 restore,
                 use_prioritize=True):
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
        self._r_heads = r_heads[1] - r_heads[
            0] + 1 if has_transform else r_heads
        self._v_heads = v_heads[1] - v_heads[
            0] + 1 if has_transform else v_heads
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
        self._use_prioritize = use_prioritize
        self._rollout_steps = config['learner']['rollout_steps']
        self._batch_size = config['learner']['batch_size']
        self._r_loss = model_config['r_loss']
        self._v_loss = model_config['v_loss']
        # model
        Model = MuZeroAtari if (self._model_cls == 'MuZeroAtari') else MuZero
        self._model = Model(
            **self._model_kwargs,
            is_train=True,
        ).to(self._device)
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
        input_state = torch.rand(1, *state_shape).to(self._device)
        self._summary_writer.add_graph(self._model.module, (input_state, ))

    def log_statistics(self, replay_buffer):
        '''log statistics for recent trajectories'''
        statistics = replay_buffer.get_statistics()
        replay_buffer.reset_statistics()
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('game/num_games', statistics.num_games, step)
        writer.add_scalar('game/num_states', statistics.num_states, step)
        game_steps = statistics.game_steps
        if game_steps:
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

    def train(self, dataloader, replay_buffer):
        '''distributed training wrapper'''
        self._train(dataloader, replay_buffer)
        self.iteration += 1

    def _train(self, dataloader, replay_buffer):
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
        print(f'>> Priority Mean: {replay_buffer.get_mean_weight():.5f}')
        if self._model_cls == 'MuZeroAtari':
            value_transform = lambda x: MuZeroAtari.to_scalar(
                self._model.v_supp, x)
            reward_transform = lambda x: MuZeroAtari.to_scalar(
                self._model.r_supp, x)
        else:
            value_transform = lambda x: x
            reward_transform = lambda x: x
        # logging
        value_target, value_rollout = [], []
        policy_target, policy_rollout = [], []
        reward_sum_target, reward_sum_rollout = [], []
        priority_sampled = []
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
            target_r_sum = torch.zeros((num_states, 1), device=self._device)
            nstep_r_sum = torch.zeros((num_states, 1), device=self._device)
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
                if t == 0 and self._use_prioritize:
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
                    nstep_r_sum[rollout_index] += reward_transform(
                        reward.detach())
                    target_r_sum[rollout_index] += reward_transform(
                        target_reward)
                    # logging
                    policy_target.extend(target_policy.tolist())
                    policy_rollout.extend(policy.detach().tolist())
                    if t == 0 and self._num_player == 1:
                        print('>>> avg target_p:', [
                            round(p, 3)
                            for p in torch.mean(target_policy, dim=0).tolist()
                        ], 'avg pred_p:', [
                            round(p, 3)
                            for p in torch.mean(policy, dim=0).tolist()
                        ])
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
            # logging
            value_target.extend(target_v_info.tolist())
            value_rollout.extend(nstep_v_sum.tolist())
            reward_sum_rollout.extend(nstep_r_sum.tolist())
            reward_sum_target.extend(target_r_sum.tolist())
            priority_sampled.extend((replay_buffer.get_mean_weight() /
                                     to_tensor(rollout.weight)).tolist())
            print(
                '... p_loss: {:.3f}, v_loss: {:.3f}, r_loss: {:.3f}, priority: {:.3f} \u00b1 {:.3f}'
                .format(
                    p_loss, v_loss, r_loss,
                    torch.mean(replay_buffer.get_mean_weight() /
                               to_tensor(rollout.weight)),
                    torch.std(replay_buffer.get_mean_weight() /
                              to_tensor(rollout.weight))))
            print(
                '... target_v: {:.3f} \u00b1 {:.3f}, rollout_v: {:.3f} \u00b1 {:.3f},'
                ' target_r_sum: {:.3f} \u00b1 {:.3f}, rollout_r_sum: {:.3f} \u00b1 {:.3f}'
                .format(
                    torch.mean(target_v_info.detach()),
                    torch.std(target_v_info.detach()),
                    torch.mean(nstep_v_sum.detach()),
                    torch.std(nstep_v_sum.detach()),
                    torch.mean(target_r_sum),
                    torch.std(target_r_sum),
                    torch.mean(nstep_r_sum),
                    torch.std(nstep_r_sum),
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
            del scalar_v, nstep_v, nstep_v_sum, ksteps, rollout_index, target_v_info
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
        get_logging_info = lambda x: dict(
            mean=np.mean(x), min=np.min(x), max=np.max(x), std=np.std(x))
        writer.add_scalars('train/value_target',
                           get_logging_info(value_target), step)
        writer.add_scalars('train/value_rollout',
                           get_logging_info(value_rollout), step)
        writer.add_scalars('log/policy_target', {
            f'action {i}': np.mean(p)
            for i, p in enumerate(zip(*policy_target))
        }, step)
        writer.add_scalars('log/policy_rollout', {
            f'action {i}': np.mean(p)
            for i, p in enumerate(zip(*policy_rollout))
        }, step)
        writer.add_scalars('train/reward_target',
                           get_logging_info(reward_sum_target), step)
        writer.add_scalars('train/reward_rollout',
                           get_logging_info(reward_sum_rollout), step)
        writer.add_scalars('train/priority',
                           get_logging_info(priority_sampled), step)
        # write back priorities
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
        subprocess.run(args, check=True, env=os.environ.copy())
        if checkpoint:
            # update the latest checkpoint
            latest_ckpt = self._ckpt_dir / 'latest.pt.zst'
            temp_ckpt = self._ckpt_dir / 'latest-temp.pt.zst'
            os.symlink(ckpt_path, temp_ckpt)
            os.replace(temp_ckpt, latest_ckpt)
