'''CZF Trainer'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from czf.learner.dataloader import RolloutBatch
from czf.learner.nn import MuZero


class Trainer:
    '''Trainer'''
    def __init__(self, args, config, checkpoint_path, model_path, log_path):
        self._device = 'cuda'
        self.model_name, self.iteration = 'default', 0
        self._ckpt_dir = checkpoint_path / self.model_name
        self._model_dir = model_path / self.model_name
        for path in (self._ckpt_dir, self._model_dir):
            path.mkdir(parents=True, exist_ok=True)
        # game
        observation_shape = config['game']['observation_shape']
        action_dim = config['game']['actions']
        # model
        model_config = config['model']
        h_channels = model_config['h_channels']
        self._model = MuZero(
            observation_shape=observation_shape,
            action_dim=action_dim,
            h_blocks=model_config['h_blocks'],
            h_channels=h_channels,
            g_blocks=model_config['g_blocks'],
            r_heads=model_config.get('r_heads', 1),
            f_blocks=model_config['f_blocks'],
            f_channels=model_config['f_channels'],
            v_heads=model_config['v_heads'],
        ).to(self._device)
        _, height, width = observation_shape
        self._input_obs = torch.rand(1, *observation_shape).to(self._device)
        self._input_state = torch.rand(1, h_channels, height,
                                       width).to(self._device)
        self._input_action = torch.rand(1, 1).to(self._device)
        # optimizer
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
        self._summary_writer = SummaryWriter(log_dir=log_path)
        # restore the latest model
        if args.restore:
            state_dict = torch.load(self._ckpt_dir / 'latest.pt')
            self.iteration = state_dict['iteration']
            self._model.load_state_dict(state_dict['model'])
            self._optimizer.load_state_dict(state_dict['optimizer'])

    def train(self, replay_buffer):
        '''optimize the model and increment model version'''
        p_criterion = lambda target_policy, policy: (
            (-target_policy * (1e-8 + policy).log()).sum(dim=1).mean())
        v_criterion = torch.nn.MSELoss()
        r_criterion = torch.nn.MSELoss()
        states_to_train = int(
            len(replay_buffer) / self._replay_retention *
            self._replay_buffer_reuse)
        dataloader = DataLoader(
            dataset=replay_buffer,
            batch_size=self._batch_size,
            collate_fn=RolloutBatch,
            shuffle=True,
        )
        self._model.train()
        gradient_scale = 1 / float(self._rollout_steps)
        num_trained_states = 0
        for rollout in dataloader:
            observation = rollout.observation
            transition = rollout.transition
            num_trained_states += len(observation)
            # prepare inputs
            observation = observation.to(self._device)
            transition = [t.to(self._device) for t in transition]
            # forward
            state = self._model.forward_representation(observation)
            loss = 0
            p_loss, v_loss, r_loss = [], [], []
            for i in range(0, len(transition), 5):
                target_policy, target_value, mask, *action_reward = transition[
                    i:i + 5]
                policy, value = self._model.forward(state)
                state.register_hook(lambda grad: grad * .5)
                state = state[mask.nonzero(as_tuple=True)]
                if action_reward:
                    action, _ = action_reward
                    state, reward = self._model.forward_dynamics(state, action)
                # loss
                p_loss_i = p_criterion(target_policy, policy)
                v_loss_i = v_criterion(target_value, value)
                loss_i = p_loss_i + v_loss_i
                p_loss.append(p_loss_i.item())
                v_loss.append(v_loss_i.item())
                if action_reward:
                    _, target_reward = action_reward
                    r_loss_i = r_criterion(target_reward, reward)
                    r_loss.append(r_loss_i.item())
                    loss_i += r_loss_i
                #print('step: {:2d}, policy loss: {:.3f}, value loss: {:.3f}'.
                #      format(i // 5, p_loss_i.item(), v_loss_i.item()))
                loss_i.register_hook(lambda grad: grad * gradient_scale)
                loss += loss_i
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
                writer.add_scalar('loss/policy', np.mean(p_loss), step)
                writer.add_scalar('loss/value', np.mean(v_loss), step)
                writer.add_scalar('loss/reward', np.mean(r_loss), step)
                break

        self.iteration += 1

    def save_model(self, checkpoint=False):
        '''save model to file'''
        model_path = self._model_dir / f'{self.iteration:05d}.pt'
        frozen_net = torch.jit.trace_module(
            self._model, {
                'forward_representation': (self._input_obs, ),
                'forward_dynamics': (self._input_state, self._input_action),
                'forward': (self._input_state, ),
            })
        frozen_net.save(str(model_path))
        os.symlink(model_path, self._model_dir / 'latest-temp.pt')
        os.replace(self._model_dir / 'latest-temp.pt',
                   self._model_dir / 'latest.pt')
        if checkpoint:
            ckpt_path = self._ckpt_dir / f'{self.iteration:05d}.pt'
            torch.save(
                {
                    'name': self.model_name,
                    'iteration': self.iteration,
                    'model': self._model.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, ckpt_path)
