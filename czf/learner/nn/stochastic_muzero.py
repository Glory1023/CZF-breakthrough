'''StochasticMuZero Model'''
from torch import nn
import torch

from czf.learner.nn import ResNet


class StochasticMuZero(nn.Module):
    '''StochasticMuZero'''
    def __init__(
        self,
        observation_shape,
        state_shape,
        action_dim,
        chance_outcome_dim,
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
        # e: encoder
        e_blocks,
        e_channels,
        # train or eval mode
        is_train,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.chance_outcome_dim = chance_outcome_dim
        # channels, height, width
        in_channels, _, _ = self.observation_shape
        _, height, width = self.state_shape
        self.representation = ResNet(in_channels, h_blocks, h_channels)
        self.prediction = ResNet(h_channels, f_blocks, f_channels)
        self.afterstate_dynamics = ResNet(h_channels + action_dim, g_blocks, h_channels)
        self.afterstate_prediction = ResNet(h_channels, f_blocks, f_channels)
        self.dynamics = ResNet(h_channels + chance_outcome_dim, g_blocks, h_channels)
        self.encoder = ResNet(in_channels + action_dim + in_channels, e_blocks, e_channels)
        # g: action plane map
        action_map = torch.zeros(action_dim, action_dim, height, width)
        for i in range(action_dim):
            action_map[i][i] = 1
        self.register_buffer('action_map', action_map)
        # i: chance_outcome plane map
        chance_outcome_map = torch.zeros(chance_outcome_dim, chance_outcome_dim, height, width)
        for i in range(chance_outcome_dim):
            chance_outcome_map[i][i] = 1
        self.register_buffer('chance_outcome_map', chance_outcome_map)
        # g => reward head
        self.reward_head_front = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1, track_running_stats=False),
            nn.ReLU(),
        )
        self.reward_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=h_channels),
            nn.ReLU(),
            nn.Linear(in_features=h_channels, out_features=r_heads[1] - r_heads[0] + 1),
            nn.Softmax(dim=1),
        )
        self.register_buffer('r_supp', torch.arange(r_heads[0], r_heads[1] + 1))
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
            nn.Linear(in_features=f_channels, out_features=v_heads[1] - v_heads[0] + 1),
            nn.Softmax(dim=1),
        )
        self.register_buffer('v_supp', torch.arange(v_heads[0], v_heads[1] + 1))
        # i => chance_outcome head
        self.chance_outcome_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2, track_running_stats=False),
            nn.ReLU(),
        )
        self.chance_outcome_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width, out_features=chance_outcome_dim),
            nn.Softmax(dim=1),
        )
        self.afterstate_value_head_front = nn.Sequential(
            nn.Conv2d(in_channels=f_channels, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1, track_running_stats=False),
            nn.ReLU(),
        )
        self.afterstate_value_head_end = nn.Sequential(
            nn.Linear(in_features=height * width, out_features=f_channels),
            nn.ReLU(),
            nn.Linear(in_features=f_channels, out_features=v_heads[1] - v_heads[0] + 1),
            nn.Softmax(dim=1),
        )
        # e => encoder
        self.encoder_head_front = nn.Sequential(
            nn.Conv2d(in_channels=e_channels, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(num_features=2, track_running_stats=False),
            nn.ReLU(),
        )
        self.encoder_head_end = nn.Sequential(
            nn.Linear(in_features=2 * height * width, out_features=chance_outcome_dim), )
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

    def forward_dynamics(self, afterstate, chance_outcome):
        '''g: dynamics function'''
        # chance_outcome is a scalar
        _, height, width = self.state_shape
        chance_outcome_plane = self.chance_outcome_map[chance_outcome.long()].view(
            -1, self.chance_outcome_dim, height, width)

        afterstate = torch.cat((afterstate, chance_outcome_plane), 1)
        x = self.dynamics(afterstate)
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
            r = StochasticMuZero.to_scalar(self.r_supp, r)
        return x, r

    def forward_dynamics_onehot(self, afterstate, chance_outcome):
        '''g: dynamics function'''
        # chance_outcome is a one-hot vector
        _, height, width = self.state_shape

        afterstate = torch.cat(
            (afterstate, chance_outcome.view(-1, self.chance_outcome_dim, 1, 1).repeat(
                1, 1, height, width)), 1)
        x = self.dynamics(afterstate)
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
            r = StochasticMuZero.to_scalar(self.r_supp, r)
        return x, r

    def forward_afterstate_dynamics(self, state, action):
        '''g: dynamics function'''
        _, height, width = self.state_shape
        action_plane = self.action_map[action.long()].view(-1, self.action_dim, height, width)

        state = torch.cat((state, action_plane), 1)
        x = self.afterstate_dynamics(state)
        x_max = x.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        x_scale = x_max - x_min
        x_scale[x_scale < 1e-6] += 1e-6
        x = (x - x_min) / x_scale
        return x

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
            v = StochasticMuZero.to_scalar(self.v_supp, v)
        return p, v

    def forward_afterstate_prediction(self, afterstate):
        '''f: prediction function'''
        _, height, width = self.state_shape
        x = self.afterstate_prediction(afterstate)
        # policy head
        c = self.chance_outcome_head_front(x)
        c = c.view(-1, 2 * height * width)
        c = self.chance_outcome_head_end(c)
        # value head
        q = self.afterstate_value_head_front(x)
        q = q.view(-1, height * width)
        q = self.afterstate_value_head_end(q)
        if not self._is_train:
            q = StochasticMuZero.to_scalar(self.v_supp, q)
        return c, q

    def forward_encoder(self, current_observation, action, next_observation):
        _, height, width = self.observation_shape
        action_plane = self.action_map[action.long()].view(-1, self.action_dim, height, width)
        # print('current observation: ', current_observation)
        # print('action plane: ', action_plane)
        # print('next observation: ', next_observation)
        obs = torch.cat((current_observation, action_plane, next_observation), 1)
        x = self.encoder(obs)
        # encoder head
        c = self.encoder_head_front(x)
        c = c.view(-1, 2 * height * width)
        logits = self.encoder_head_end(c)
        # chance_probs = torch.nn.functional.gumbel_softmax(logits, tau=1e-5, hard=False)
        # chance_outcome = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True)
        # print('chance_probs: ', chance_probs)
        # print('chance_outcome: ', chance_outcome)
        # return chance_probs, chance_outcome

        y_soft = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False)
        dim = -1
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft

        return ret, y_soft

    @staticmethod
    def to_scalar(supp, x, epsilon=0.001):
        '''distribution to scalar'''
        batch = x.size(0)
        x_supp = supp.expand(batch, -1)
        x = (x_supp * x).sum(1, keepdim=True)
        # inverse transform
        # sign(x) * (((sqrt(1 + 4 * eps * (|x| + 1 + eps) - 1) / (2 * eps)) ** 2 - 1)
        return torch.sign(x) * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) /
                                 (2 * epsilon))**2 - 1)