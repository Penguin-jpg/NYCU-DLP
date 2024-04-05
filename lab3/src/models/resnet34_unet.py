# Implement your ResNet34_UNet model here
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # conv(k=1, s=1 or 2, p=1) -> bn -> relu -> conv(k=1, s=1, p=1) -> bn ->
        # residual -> relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # because the number of channels maybe different for in_channels and
        # out_channels, the shortcut needs to be a 1x1 conv to match the number of channels
        if in_channels != out_channels or stride == 2:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(x)
        h = self.relu(x)
        h = self.conv2(x)
        h = self.bn2(x)
        x = self.shortcut(x)
        h += x
        h = self.relu(h)
        return h


class ResNet34(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(ResNet34).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(64, 64, stride=1),
            ),
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1),
        )

        self.conv3 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),
            ResBlock(128, 128, stride=1),
            ResBlock(128, 128, stride=1),
        )

        self.conv4 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
            ResBlock(256, 256, stride=1),
        )

        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


assert False, "Not implemented yet!"
