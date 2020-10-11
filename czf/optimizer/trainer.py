'''CZF Trainer'''
import os
import torch

from czf.optimizer.nn import AlphaZero


class Trainer:
    '''Trainer'''
    def __init__(self, args, config, model_path, traced_model_path):
        self.device = 'cuda'
        self.model_name, self.model_version = 'default', 0
        self.model_dir = model_path / self.model_name
        self.traced_model_dir = traced_model_path / self.model_name
        for path in (self.model_dir, self.traced_model_dir):
            path.mkdir(parents=True, exist_ok=True)
        # game (TODO)
        observation_tensor_shape = (4, 4, 4)
        action_dim = 9
        self.inputs = torch.rand(1, *observation_tensor_shape).to(self.device)
        # model
        model_config = config['model']
        self.model = AlphaZero(
            observation_tensor_shape=observation_tensor_shape,
            action_dim=action_dim,
            channels=model_config.get('channels', 128),
            blocks=model_config.get('blocks', 128),
            v_heads=model_config.get('v_heads', 1),
        ).to(self.device)
        # restore the latest model
        if args.restore:
            state_dict = torch.load(self.model_dir / 'latest.pt')
            self.model_version = state_dict['version']
            self.model.load_state_dict(state_dict['model'])

    def save_model(self, trace=False):
        '''save model to file'''
        model_path = self.model_dir / f'{self.model_version:05d}.pt'
        torch.save(
            {
                'name': self.model_name,
                'version': self.model_version,
                'model': self.model.state_dict(),
            }, model_path)
        if trace:
            traced_model_path = self.traced_model_dir / f'{self.model_version:05d}.pt'
            frozen_net = torch.jit.trace(self.model, self.inputs)
            frozen_net.save(str(traced_model_path))
            os.symlink(traced_model_path,
                       self.traced_model_dir / 'latest-temp.pt')
            os.replace(self.traced_model_dir / 'latest-temp.pt',
                       self.traced_model_dir / 'latest.pt')
