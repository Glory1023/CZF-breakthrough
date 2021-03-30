'''AlphaZero Model'''
from torch import nn

from czf.learner.nn import ResNet, SEResNet


class AlphaZero(nn.Module):
    '''AlphaZero'''
    def __init__(
        self,
        observation_tensor_shape,
        action_dim,
        channels,
        blocks,
        v_heads,
        backbone,
        fc_hidden_dimension=16
    ):
        super().__init__()
        self.observation_tensor_shape = observation_tensor_shape
        # channels, height, width
        in_channels, height, width = self.observation_tensor_shape

        if backbone == "SE-ResNet":
            self.backbone = SEResNet(in_channels, blocks, channels, fc_hidden_dimension)
        else:
            self.backbone = ResNet(in_channels, blocks, channels)

        # policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
        )
        self.policy_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width, out_features=action_dim),
            nn.Softmax(dim=1)
        )

        # value head
        self.value_head_front = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )
        self.value_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=channels),
            nn.ReLU(),
            nn.Linear(in_features=channels, out_features=v_heads),
            nn.Tanh()
        )

    def forward(self, x):
        '''forward'''
        _, height, width = self.observation_tensor_shape
        x = self.backbone(x)
        # policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * height * width)
        p = self.policy_head_end(p)
        # value head
        v = self.value_head_front(x)
        v = v.view(-1, height * width)
        v = self.value_head_end(v)
        return p, v
