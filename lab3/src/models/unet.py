# Implement your UNet model here
import torch
from torch import nn
from torchvision.transforms.functional import center_crop
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # basic block of unet is 2 convolution layers and 2 relu
        # Kaiming initialization is better than the original paper's initialization
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # nn.init.normal_(conv1.weight, mean=0, std=math.sqrt(2/(3*3*in_channels)))
        nn.init.kaiming_normal_(conv1.weight, mode="fan_in", nonlinearity="relu")
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # nn.init.normal_(conv2.weight, mean=0, std=math.sqrt(2/(3*3*out_channels)))
        nn.init.kaiming_normal_(conv2.weight, mode="fan_in", nonlinearity="relu")

        self.block = nn.Sequential(
            # I choose to maintain image size or the preprocessing of mask will need to be deal with separately
            # 3x3 conv with stride 1 and padding 1 to maintain image size
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_multipliers=[1, 2, 4, 8]):
        super(Encoder, self).__init__()

        # encoder downsamples the image 2x for every block
        # we also need to store the convolution results for skip-connection
        self.blocks = nn.ModuleList([])
        for multiplier in channel_multipliers:
            # the output channels are is the base_channels * the multiplier
            out_channels = base_channels * multiplier

            # append block to encoder
            self.blocks.append(ConvBlock(in_channels, out_channels))
            self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # update input channels for next iteration
            in_channels = out_channels

    def forward(self, x):
        encoder_results = []

        for module in self.blocks:
            # store the result if the current module is ConvBlock
            if isinstance(module, ConvBlock):
                x = module(x)
                # print(f"conv: {x.shape}")
                encoder_results.append(x)
            else:
                x = module(x)
                # print(f"pool: {x.shape}")
        return x, encoder_results


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, base_channels=64, channel_multipliers=[8, 4, 2, 1]):
        super(Decoder, self).__init__()

        # decoder upsamples the image 2x for every block
        # we also need to concat the convolution results from encoder for skip-connection
        self.blocks = nn.ModuleList([])
        for i, multiplier in enumerate(channel_multipliers):
            # the output channels are is the base_channels * the multiplier
            out_channels = base_channels * multiplier

            # the first upsample is done in the bottleneck, so we only need to do 3 upsamples
            if i != len(channel_multipliers) - 1:
                block = nn.Sequential(
                    ConvBlock(in_channels, out_channels),
                    # 2x2 deconvolution with stride 2 and padding 2 to upsample 2x
                    # out_channels divided by because we will concat the other half
                    nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2),
                )
            else:
                block = nn.Sequential(
                    ConvBlock(in_channels, out_channels),
                )

            # append bloc to encoder
            self.blocks.append(block)

            # update input channels for next iteration
            in_channels = out_channels

        # to Sequential
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x, encoder_results):
        # reverse the encoder results because concat from back to front
        encoder_results = encoder_results[::-1]

        for block, encoder_result in zip(self.blocks, encoder_results):
            # because the image size in encoder is different from the image size in decoder,
            # we need to resize the encoder_result with center crop
            encoder_result = center_crop(encoder_result, x.shape[2:])
            # print(f"x: {x.shape} encoder_result: {encoder_result.shape}")
            # print(f"crop: {encoder_result.shape}")
            # concat encoder_result with x along the channel dimension
            x = torch.cat([encoder_result, x], dim=1)
            # print(f"concat: {x.shape}")
            x = block(x)
            # print(f"deconv: {x.shape}")
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
    ):
        super(UNet, self).__init__()

        # unet consists of an encoder, a bottleneck, and a decoder
        self.encoder = Encoder(in_channels, base_channels, channel_multipliers)
        in_channels = base_channels * channel_multipliers[-1]

        # bottleneck consists of 2 convolution layers and a deconvolution layer: 512 -> 1024 -> 1024 -> upsample
        self.bottleneck = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2),
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=2, stride=2),
        )
        in_channels *= 2

        # since we need to upsample, we have to reverse the channel multipliers
        self.decoder = Decoder(in_channels, base_channels, channel_multipliers[::-1])

        # output layer is a convoluton layer with 1x1 kernel
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.normal_(self.output.weight, mean=0, std=math.sqrt(2 / (1 * 1 * base_channels)))

    def forward(self, x):
        x, encoder_results = self.encoder(x)
        x = self.bottleneck(x)
        # print(f"bottleneck: {x.shape}")
        x = self.decoder(x, encoder_results)
        x = self.output(x)
        # print(f"output: {x.shape}")
        return x


# unet = UNet(in_channels=3, out_channels=3, base_channels=64, channel_multipliers=[1, 2, 4, 8])
# t = torch.randn(1, 2, 572, 572)
# unet(t)
# print(unet)
