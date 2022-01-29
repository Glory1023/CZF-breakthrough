from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, fc_hidden_dimension=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, fc_hidden_dimension, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_dimension, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fc_hidden_dimension=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.se = SELayer(out_channels, fc_hidden_dimension)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.se(y)
        x = x + y
        x = self.relu2(x)
        return x


class SEResNet(nn.Module):
    def __init__(self, in_channels, blocks, out_channels, fc_hidden_dimension=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            SEBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                fc_hidden_dimension=fc_hidden_dimension,
            ) for _ in range(blocks)
        ])

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.convs:
            x = conv(x)
        return x
