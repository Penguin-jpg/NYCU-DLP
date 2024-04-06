# Implement your ResNet34_UNet model here
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # conv(k=1, s=1 or 2, p=1) -> bn -> relu -> conv(k=1, s=1, p=1) -> bn ->
        # residual -> relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        x = self.shortcut(x)
        h += x
        h = self.relu(h)
        return h


class ResNet34(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
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

        # we don't need to classify, so we can stop here
        self.conv5 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),
            ResBlock(512, 512, stride=1),
        )

    def forward(self, x):
        encoder_results = []
        x = self.conv1(x)
        encoder_results.append(x)
        # print(f"conv1: {x.shape}")
        x = self.conv2(x)
        encoder_results.append(x)
        # print(f"conv2: {x.shape}")
        x = self.conv3(x)
        encoder_results.append(x)
        # print(f"conv3: {x.shape}")
        x = self.conv4(x)
        encoder_results.append(x)
        # print(f"conv4: {x.shape}")
        x = self.conv5(x)
        # print(f"conv5: {x.shape}")
        return x, encoder_results


# because the architecture is different from how I write UNet, I will use create
# a new block here
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, encoder_result=None):
        # upsample x
        x = self.up(x)
        # concate x and encoder_result along the channel dimension
        if encoder_result is not None:
            x = torch.cat([encoder_result, x], dim=1)
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_result_channels=[256, 128, 64, 64]):
        super(Decoder, self).__init__()

        in_channels = 512 + encoder_result_channels[0]
        out_channels = encoder_result_channels[0]
        blocks = nn.ModuleList([UpBlock(in_channels, out_channels)])
        for i in range(1, 4):
            in_channels = out_channels + encoder_result_channels[i]
            if i == 3:
                out_channels = 32
            else:
                out_channels = encoder_result_channels[i]
            blocks.append(UpBlock(in_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, encoder_results):
        # reverse the encoder results because concat from back to front
        encoder_results = encoder_results[::-1]

        for block, encoder_result in zip(self.blocks, encoder_results):
            x = block(x, encoder_result)
            # print(f"decoder: {x.shape}")
        return x


# reference: https://www.researchgate.net/figure/UNet-architecture-with-a-ResNet-34-encoder-The-output-of-the-additional-1x1-convolution_fig3_350858002
class ResNet34Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(ResNet34Unet, self).__init__()

        self.encoder = ResNet34(in_channels)

        self.decoder = Decoder([256, 128, 64, 64])

        self.output = nn.Sequential(
            UpBlock(32, 16),
            nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x, encoder_results = self.encoder(x)
        x = self.decoder(x, encoder_results)
        x = self.output(x)
        # print(f"output: {x.shape}")
        return x


# net = ResNet34Unet(in_channels=3, out_channels=2)
# t = torch.randn(1, 3, 256, 256)
# print(net(t).shape)
