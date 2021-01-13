'''MuZeroAtari Model'''
from torch import nn
import torch

from czf.learner.nn import ResNet


class MuZeroAtari(nn.Module):
    '''MuZeroAtari'''
    def __init__(
        self,
        observation_shape,
        state_shape,
        action_dim,
        # h: representation function
        h_blocks,
        h_channels,
        # g: dynamics function
        g_blocks,
        # reward support: [r_low, r_high]
        r_heads,
        # f: prediction function
        f_blocks,
        f_channels,
        # value support: [v_low, v_high]
        v_heads,
        # train or eval mode
        is_train,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        # channels, height, width
        in_channels, _, _ = self.observation_shape
        _, height, width = self.state_shape
        self.representation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=h_channels // 2,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            ResNet(in_channels=h_channels // 2,
                   blocks=1,
                   out_channels=h_channels // 2),
            nn.Conv2d(in_channels=h_channels // 2,
                      out_channels=h_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            ResNet(in_channels=h_channels,
                   blocks=h_blocks,
                   out_channels=h_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ResNet(in_channels=h_channels,
                   blocks=h_blocks,
                   out_channels=h_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.dynamics = ResNet(h_channels + 1, g_blocks, h_channels)
        self.prediction = ResNet(h_channels, f_blocks, f_channels)
        # g: action plane map
        action_map = torch.cat([
            torch.full((1, *state_shape[-2:]), (i + 1) / action_dim)
            for i in range(action_dim)
        ])
        self.register_buffer('action_map', action_map)
        # g => reward head
        self.reward_head_front = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1, track_running_stats=False),
            nn.ReLU(),
        )
        self.reward_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=h_channels),
            nn.ReLU(),
            nn.Linear(in_features=h_channels,
                      out_features=r_heads[1] - r_heads[0] + 1),
            nn.Softmax(dim=1),
        )
        self.register_buffer('r_supp', torch.arange(r_heads[0],
                                                    r_heads[1] + 1))
        # f => policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2, track_running_stats=False),
            nn.ReLU(),
        )
        self.policy_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width, out_features=action_dim),
            nn.Softmax(dim=1),
        )
        # f => value head
        self.value_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1, track_running_stats=False),
            nn.ReLU(),
        )
        self.value_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=f_channels),
            nn.ReLU(),
            nn.Linear(in_features=f_channels,
                      out_features=v_heads[1] - v_heads[0] + 1),
            nn.Softmax(dim=1),
        )
        self.register_buffer('v_supp', torch.arange(v_heads[0],
                                                    v_heads[1] + 1))
        self.epsilon = 0.001
        self._is_train = is_train

    def forward_representation(self, observation):
        '''h: representation function'''
        x = self.representation(observation)
        x_max = x.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        x_scale = x_max - x_min
        x_scale[x_scale < 1e-6] += 1e-6
        x = (x - x_min) / x_scale
        return x

    def forward_dynamics(self, state, action):
        '''g: dynamics function'''
        _, height, width = self.state_shape
        action_plane = self.action_map[action.long()].view(
            -1, 1, height, width)
        state = torch.cat((state, action_plane), 1)
        x = self.dynamics(state)
        x_max = x.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        x_scale = x_max - x_min
        x_scale[x_scale < 1e-6] += 1e-6
        x = (x - x_min) / x_scale
        # reward head
        r = self.reward_head_front(x)
        r = r.view(-1, height * width)
        r = self.reward_head_end(r)
        if not self._is_train:
            r = MuZeroAtari.to_scalar(self.r_supp, r)
        return x, r

    def forward(self, state):
        '''f: prediction function'''
        _, height, width = self.state_shape
        x = self.prediction(state)
        # policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * height * width)
        p = self.policy_head_end(p)
        # value head
        v = self.value_head_front(x)
        v = v.view(-1, height * width)
        v = self.value_head_end(v)
        if not self._is_train:
            v = MuZeroAtari.to_scalar(self.v_supp, v)
        return p, v

    @staticmethod
    def to_scalar(supp, x, epsilon=0.001):
        '''distribution to scalar'''
        batch = x.size(0)
        x_supp = supp.expand(batch, -1)
        x = (x_supp * x).sum(1, keepdim=True)
        # inverse transform
        # sign(x) * (((sqrt(1 + 4 * eps * (|x| + 1 + eps) - 1) / (2 * eps)) ** 2 - 1)
        return torch.sign(x) * (
            ((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) /
             (2 * epsilon))**2 - 1)
