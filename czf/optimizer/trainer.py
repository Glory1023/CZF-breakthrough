'''CZF Trainer'''
import os
import torch
from torch.utils.data import DataLoader

from czf.optimizer.nn import MuZero


class Trainer:
    '''Trainer'''
    def __init__(self, args, config, model_path, traced_model_path):
        self._device = 'cuda'
        self.model_name, self.model_version = 'default', 0
        self._model_dir = model_path / self.model_name
        self._traced_model_dir = traced_model_path / self.model_name
        for path in (self._model_dir, self._traced_model_dir):
            path.mkdir(parents=True, exist_ok=True)
        # game
        observation_tensor_shape = config['game']['observation_shape']
        action_dim = config['game']['actions']
        # model
        model_config = config['model']
        f_in_channels = model_config.get('f_in_channels', 128)
        self._model = MuZero(
            observation_tensor_shape=observation_tensor_shape,
            action_dim=action_dim,
            h_blocks=model_config.get('h_blocks', 128),
            g_blocks=model_config.get('g_blocks', 128),
            f_in_channels=f_in_channels,
            f_channels=model_config.get('f_channels', 128),
            f_blocks=model_config.get('f_blocks', 128),
            v_heads=model_config.get('v_heads', 1),
        ).to(self._device)
        self._input_obs = torch.rand(1, *observation_tensor_shape).to(
            self._device)
        self._input_state = torch.rand(
            f_in_channels, *observation_tensor_shape).to(self._device)
        self._input_action = torch.rand(1, 1).to(self._device)
        # optimizer
        self._replay_buffer_reuse = config['optimizer']['replay_buffer_reuse']
        self._replay_retention = config['optimizer'][
            'replay_buffer_size'] / config['optimizer']['frequency']
        self._batch_size = config['optimizer']['batch_size']
        self._optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=config['optimizer']['learning_rate'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay'],
            nesterov=config['optimizer']['nesterov'],
        )
        # restore the latest model
        if args.restore:
            state_dict = torch.load(self._model_dir / 'latest.pt')
            self.model_version = state_dict['version']
            self._model.load_state_dict(state_dict['model'])

    def train(self, replay_buffer):
        '''optimize the model and increment model version'''
        p_criterion = lambda p_targets, p_logits: ((
            -p_targets * torch.log_softmax(p_logits, dim=1)).sum(dim=1).mean())
        v_criterion = lambda v_targets, v_logits: ((
            -v_targets * torch.log_softmax(v_logits, dim=1)).sum(dim=1).mean())
        #policy_loss = (-target_policy * (1e-8 + policy).log()).sum(dim=1).mean()
        #value_loss = torch.nn.MSELoss()(target_value, value)
        states_to_train = int(
            len(replay_buffer) / self._replay_retention *
            self._replay_buffer_reuse)
        dataloader = DataLoader(
            dataset=replay_buffer,
            batch_size=self._batch_size,
            shuffle=True,
        )
        self._model.train()
        num_trained_states = 0
        for observation_tensor, target_policy, target_value in dataloader:
            num_trained_states += len(observation_tensor)
            # the last batch might need to drop some inputs
            drop = num_trained_states - states_to_train
            if drop > 0:
                num_trained_states -= drop
                observation_tensor = observation_tensor[drop:]
                target_policy = target_policy[drop:]
                target_value = target_value[drop:]
            if len(observation_tensor) == 0:
                break
            # prepare inputs
            observation_tensor = observation_tensor.to(self._device)
            target_policy = target_policy.to(self._device)
            target_value = target_value.to(self._device)
            # forward
            policy, value = self._model.forward(observation_tensor)
            policy_loss = p_criterion(target_policy, policy)
            value_loss = p_criterion(target_value, value)
            loss = policy_loss + value_loss
            # optimize
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if num_trained_states >= states_to_train:
                break

        self.model_version += 1

    def save_model(self, trace=False):
        '''save model to file'''
        model_path = self._model_dir / f'{self.model_version:05d}.pt'
        torch.save(
            {
                'name': self.model_name,
                'version': self.model_version,
                'model': self._model.state_dict(),
            }, model_path)
        if trace:
            traced_model_path = self._traced_model_dir / f'{self.model_version:05d}.pt'
            frozen_net = torch.jit.trace_module(
                self._model, {
                    'forward_representation': (self._input_obs, ),
                    'forward_dynamics':
                    (self._input_state, self._input_action),
                    'forward': (self._input_state, ),
                })
            frozen_net.save(str(traced_model_path))
            os.symlink(traced_model_path,
                       self._traced_model_dir / 'latest-temp.pt')
            os.replace(self._traced_model_dir / 'latest-temp.pt',
                       self._traced_model_dir / 'latest.pt')
