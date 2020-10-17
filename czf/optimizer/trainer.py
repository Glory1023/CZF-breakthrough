'''CZF Trainer'''
import os
import torch
from torch.utils.data import DataLoader

from czf.optimizer.dataloader import RolloutBatch
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
        observation_shape = config['game']['observation_shape']
        action_dim = config['game']['actions']
        # model
        model_config = config['model']
        h_channels = model_config.get('h_channels', 128)
        self._model = MuZero(
            observation_shape=observation_shape,
            action_dim=action_dim,
            h_blocks=model_config.get('h_blocks', 3),
            h_channels=h_channels,
            g_blocks=model_config.get('g_blocks', 2),
            r_heads=model_config.get('r_heads', 1),
            f_blocks=model_config.get('f_blocks', 3),
            f_channels=model_config.get('f_channels', 128),
            v_heads=model_config.get('v_heads', 1),
        ).to(self._device)
        _, height, width = observation_shape
        self._input_obs = torch.rand(1, *observation_shape).to(self._device)
        self._input_state = torch.rand(1, h_channels, height,
                                       width).to(self._device)
        self._input_action = torch.rand(1, 1).to(self._device)
        # optimizer
        self._replay_buffer_reuse = config['optimizer']['replay_buffer_reuse']
        self._replay_retention = config['optimizer'][
            'replay_buffer_size'] / config['optimizer']['frequency']
        self._rollout_steps = config['optimizer']['rollout_steps']
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
                p_loss = p_criterion(target_policy, policy)
                v_loss = v_criterion(target_value, value)
                loss_i = p_loss + v_loss
                if action_reward:
                    _, target_reward = action_reward
                    r_loss = r_criterion(target_reward, reward)
                    loss_i += r_loss
                print('step: {:2d}, policy loss: {:.3f}, value loss: {:.3f}'.
                      format(i // 5, p_loss.item(), v_loss.item()))
                loss_i.register_hook(lambda grad: grad * gradient_scale)
                loss += loss_i
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
