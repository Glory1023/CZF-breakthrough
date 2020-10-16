from torch import nn
import torch

from czf.optimizer.nn import ResNet


class MuZero(nn.Module):
    def __init__(
        self,
        observation_shape,
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
    ):
        super().__init__()
        self.observation_shape = observation_shape
        # channels, height, width
        in_channels, height, width = self.observation_shape
        self.representation = ResNet(in_channels, h_blocks, h_channels)
        self.dynamics = ResNet(h_channels + 1, g_blocks, h_channels)
        self.prediction = ResNet(h_channels, f_blocks, f_channels)
        # g: board action map
        self.board = torch.nn.Parameter(torch.eye(height * width),
                                        requires_grad=False)
        # g => reward head
        self.reward_head_front = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.reward_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=h_channels),
            nn.ReLU(), nn.Linear(in_features=h_channels, out_features=r_heads),
            nn.Tanh())
        # f => policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
        )
        self.policy_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width, out_features=action_dim),
            nn.Softmax(dim=1))
        # f => value head
        self.value_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.value_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=f_channels),
            nn.ReLU(), nn.Linear(in_features=f_channels, out_features=v_heads),
            nn.Tanh())

    def forward_representation(self, observation):
        x = self.representation(observation)
        return x

    def forward_dynamics(self, state, action):
        _, height, width = self.observation_shape
        board_action = self.board[action.long()].view(-1, 1, height, width)
        state = torch.cat((state, board_action), 1)
        x = self.dynamics(state)
        # reward head
        r = self.reward_head_front(x)
        r = r.view(-1, height * width)
        r = self.reward_head_end(r)
        return x, r

    def forward(self, state):
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
