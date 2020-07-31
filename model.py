import torch
import torch.nn as nn


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBlockDown, self).__init__()

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.left(x) + self.shortcut(x)


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBlockUp, self).__init__()

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.left(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_shortcut=False):
        super(ResidualBlock, self).__init__()

        self.use_shortcut = use_shortcut

        self.left = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.left(x) + self.shortcut(x)


class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        self.in_channel = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
        )

        self.conv2 = nn.Sequential(
            ResidualBlockDown(64, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
        )

        self.conv3 = nn.Sequential(
            ResidualBlockDown(128, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
        )

        self.conv4 = nn.Sequential(
            ResidualBlockDown(256, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
            ResidualBlock(512, 128, 512),
        )

        self.conv5 = nn.Sequential(
            ResidualBlockUp(512, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
            ResidualBlock(256, 64, 256),
        )

        self.conv6 = nn.Sequential(
            ResidualBlockUp(256, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
            ResidualBlock(128, 32, 128),
        )

        self.conv7 = nn.Sequential(
            ResidualBlockUp(128, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
            ResidualBlock(64, 16, 64),
        )

        self.conv8 = nn.Sequential(
            ResidualBlockUp(64, 16, 32),
            ResidualBlock(32, 8, 32),
            ResidualBlock(32, 8, 32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)

        u1 = d3 + self.conv5(d4)
        u2 = d2 + self.conv6(u1)
        u3 = d1 + self.conv7(u2)
        u4 = self.conv8(u3)

        return u4
