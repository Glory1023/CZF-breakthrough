'''CZF Trainer'''
from io import BytesIO
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import zstandard as zstd

from czf.learner.dataloader import RolloutBatch
from czf.learner.nn import MuZero, MuZeroAtari


class Trainer:
    '''Trainer'''
    def __init__(self,
                 config,
                 checkpoint_path,
                 model_path,
                 log_path,
                 model_name,
                 restore,
                 rank=0):
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
        action_dim = config['game']['actions']
        # model
        model_config = config['model']
        h_channels = model_config['h_channels']
        state_shape = [h_channels, *config['game']['state_spatial_shape']]
        Model = MuZeroAtari if (config['model']['name']
                                == 'MuZeroAtari') else MuZero
        # torch.cuda.set_device(rank)
        # torch.distributed.init_process_group('nccl', init_method='env://')
        self._model = Model(
            observation_shape=observation_shape,
            state_shape=state_shape,
            action_dim=action_dim,
            h_blocks=model_config['h_blocks'],
            h_channels=h_channels,
            g_blocks=model_config['g_blocks'],
            r_heads=model_config.get('r_heads', 1),
            f_blocks=model_config['f_blocks'],
            f_channels=model_config['f_channels'],
            v_heads=model_config['v_heads'],
            is_train=True,
        ).to(self._device)
        # self._model = DataParallelWrapper(self._model,
        #                                   device_ids=[rank],
        #                                   output_device=rank)
        self._input_obs = torch.rand(1, *observation_shape).to(self._device)
        self._input_state = torch.rand(1, *state_shape).to(self._device)
        self._input_action = torch.rand(1, 1).to(self._device)
        # optimizer
        torch.backends.cudnn.benchmark = True
        self._replay_buffer_reuse = config['learner']['replay_buffer_reuse']
        self._replay_retention = config['learner'][
            'replay_buffer_size'] / config['learner']['frequency']
        self._rollout_steps = config['learner']['rollout_steps']
        self._batch_size = config['learner']['batch_size']
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=config['learner']['optimizer']['learning_rate'],
            momentum=config['learner']['optimizer']['momentum'],
            weight_decay=config['learner']['optimizer']['weight_decay'],
            nesterov=config['learner']['optimizer']['nesterov'],
        )
        # compressor
        self._model_compressor = zstd.ZstdCompressor()
        self._ckpt_compressor = zstd.ZstdCompressor()
        # restore the latest checkpoint
        if restore:
            print('Restore from', restore)
            with open(restore, 'rb') as model_blob:
                dctx = zstd.ZstdDecompressor()
                model = dctx.decompress(model_blob.read())
            state_dict = torch.load(BytesIO(model))
            self.iteration = state_dict['iteration']
            self._model.load_state_dict(state_dict['model'])
            self._optimizer.load_state_dict(state_dict['optimizer'])
        # tensorboard log
        self._summary_writer = SummaryWriter(log_dir=log_path,
                                             purge_step=self.iteration)
        # note: PyTorch supports the `forward` method currently
        # so, we can only trace the prediction model now.
        self._summary_writer.add_graph(self._model, (self._input_state, ))

    def log_statistics(self, replay_buffer):
        '''log statistics for recent trajectories'''
        statistics = replay_buffer.get_statistics()
        replay_buffer.reset_statistics()
        writer, step = self._summary_writer, self.iteration
        writer.add_scalar('game/num_games', statistics.num_games, step)
        game_steps = statistics.game_steps
        writer.add_scalar('game/num_states', sum(game_steps), step)
        writer.add_scalars(
            'game/steps', {
                'mean': np.mean(game_steps),
                'min': np.min(game_steps),
                'max': np.max(game_steps),
                'std': np.std(game_steps),
            }, step)
        for p, player_returns in enumerate(statistics.player_returns):
            returns_rates = {
                key: value / statistics.num_games
                for key, value in player_returns.items()
            }
            writer.add_scalars(f'game/player{p}_rate', returns_rates, step)

    def train(self, replay_buffer):
        '''distributed training wrapper'''
        # with self._model.no_sync():
        self._train(replay_buffer)

    def _train(self, replay_buffer):
        '''optimize the model and increment model version'''
        p_criterion = lambda target_policy, policy: (
            (-target_policy * (1e-8 + policy).log()).sum(dim=1))
        v_criterion = torch.nn.MSELoss(reduction='none')
        r_criterion = torch.nn.MSELoss(reduction='none')
        scale_gradient = lambda tensor, scale: (tensor * scale + tensor.detach(
        ) * (1 - scale))
        states_to_train = int(
            len(replay_buffer) / self._replay_retention *
            self._replay_buffer_reuse)
        # sampler = DistributedSampler(replay_buffer)
        dataloader = DataLoader(
            dataset=replay_buffer,
            batch_size=self._batch_size,
            # sampler=sampler,
            shuffle=True,
            collate_fn=RolloutBatch,
            pin_memory=True,
        )
        self._model.train()
        num_trained_states = 0
        for rollout in dataloader:
            observation = rollout.observation.to(self._device)
            scale = rollout.scale.to(self._device)
            transition = [t.to(self._device) for t in rollout.transition]
            num_trained_states += len(observation)
            # forward
            state = self._model.forward_representation(observation)
            total_batch, loss, p_loss, v_loss, r_loss = 0, 0, 0, 0, 0
            for i in range(0, len(transition), 5):
                target_value, mask, target_policy, action, target_reward = transition[
                    i:i + 5]
                mask = mask.nonzero(as_tuple=True)
                total_batch += len(state)
                policy, value = self._model.forward(state)
                policy = policy[mask]
                state = state[mask]
                state, reward = self._model.forward_dynamics(state, action)
                state = scale_gradient(state, 0.5)
                # loss
                p_loss_i = p_criterion(target_policy, policy)
                v_loss_i = v_criterion(target_value, value)
                r_loss_i = r_criterion(target_reward, reward)
                # scale gradient
                if i > 0:
                    v_loss_i = scale_gradient(v_loss_i, scale.view(-1, 1))
                scale = scale[mask]
                if i > 0:
                    p_loss_i = scale_gradient(p_loss_i, scale)
                r_loss_i = scale_gradient(r_loss_i, scale.view(-1, 1))
                # total loss
                p_loss_i = p_loss_i.sum()
                v_loss_i = v_loss_i.sum()
                r_loss_i = r_loss_i.sum()
                loss_i = p_loss_i + v_loss_i + r_loss_i
                loss += loss_i
                p_loss += p_loss_i.item()
                v_loss += v_loss_i.item()
                r_loss += r_loss_i.item()
            loss /= total_batch
            p_loss /= total_batch
            v_loss /= total_batch
            r_loss /= total_batch
            print('policy loss: {:.3f}, value loss: {:.3f}'.format(
                p_loss, v_loss))
            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            # the end of current training
            if num_trained_states >= states_to_train:
                lr = next(iter(self._optimizer.param_groups))['lr']
                writer, step = self._summary_writer, self.iteration
                writer.add_scalar('params/lr', lr, step)
                writer.add_scalar('loss/', loss.item(), step)
                writer.add_scalar('loss/policy', p_loss, step)
                writer.add_scalar('loss/value', v_loss, step)
                writer.add_scalar('loss/reward', r_loss, step)
                break

        self.iteration += 1

    def save_model(self, checkpoint=False):
        '''save model to file'''
        buffer = BytesIO()
        with torch.jit.optimized_execution(True):
            frozen_net = torch.jit.trace_module(
                self._model, {
                    'forward_representation': (self._input_obs, ),
                    'forward_dynamics':
                    (self._input_state, self._input_action),
                    'forward': (self._input_state, ),
                })
            torch.jit.save(frozen_net, buffer)
        buffer.seek(0)
        compressed = self._model_compressor.compress(buffer.read())
        model_path = self._model_dir / f'{self.iteration:05d}.pt.zst'
        model_path.write_bytes(compressed)
        latest_model = self._model_dir / 'latest.pt.zst'
        temp_model = self._model_dir / 'latest-temp.pt.zst'
        os.symlink(model_path, temp_model)
        os.replace(temp_model, latest_model)
        # save a checkpoint
        if checkpoint:
            buffer = BytesIO()
            torch.save(
                {
                    'name': self.model_name,
                    'iteration': self.iteration,
                    'model': self._model.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, buffer)
            buffer.seek(0)
            compressed = self._ckpt_compressor.compress(buffer.read())
            ckpt_path = self._ckpt_dir / f'{self.iteration:05d}.pt.zst'
            ckpt_path.write_bytes(compressed)
            # update the latest checkpoint
            latest_ckpt = self._ckpt_dir / 'latest.pt.zst'
            temp_ckpt = self._ckpt_dir / 'latest-temp.pt.zst'
            if not latest_ckpt.exists():
                os.symlink(ckpt_path, latest_ckpt)
            os.symlink(ckpt_path, temp_ckpt)
            os.replace(temp_ckpt, latest_ckpt)
