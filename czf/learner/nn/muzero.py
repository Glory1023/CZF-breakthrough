'''MuZero Model'''
from torch import nn
import torch

from czf.learner.nn import ResNet


class MuZero(nn.Module):
    '''MuZero'''
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
        r_heads,
        # f: prediction function
        f_blocks,
        f_channels,
        v_heads,
        # train or eval mode pylint: disable=unused-argument
        is_train,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        assert state_shape[1:] == observation_shape[1:]
        # channels, height, width
        in_channels, height, width = self.observation_shape
        self.representation = ResNet(in_channels, h_blocks, h_channels)
        self.dynamics = ResNet(h_channels + 1, g_blocks, h_channels)
        self.prediction = ResNet(h_channels, f_blocks, f_channels)
        # g: board action map
        self.register_buffer('board', torch.eye(height * width))
        # g => reward head
        self.reward_head_front = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1, track_running_stats=False),
            nn.ReLU(),
        )
        self.reward_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=h_channels),
            nn.ReLU(),
            nn.Linear(in_features=h_channels, out_features=r_heads),
            nn.Tanh(),
        )
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
            nn.Linear(in_features=f_channels, out_features=v_heads),
            nn.Tanh(),
        )

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
        _, height, width = self.observation_shape
        board_action = self.board[action.long()].view(-1, 1, height, width)
        state = torch.cat((state, board_action), 1)
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
        return x, r

    def forward(self, state):
        '''f: prediction function'''
        _, height, width = self.observation_shape
        x = self.prediction(state)
        # policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * height * width)
        p = self.policy_head_end(p)
        # value head
        v = self.value_head_front(x)
        v = v.view(-1, height * width)
        v = self.value_head_end(v)
        return p, v
