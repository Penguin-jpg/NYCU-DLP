import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# generator and discriminator are modified from https://github.com/leo27945875/MHingeGAN-for-multi-label-conditional-generation


def linear(in_features, out_features, apply_sn=True):
    return (
        spectral_norm(nn.Linear(in_features, out_features))
        if apply_sn
        else nn.Linear(in_features, out_features)
    )


def conv_2d(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_sn=True
):
    return (
        spectral_norm(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        )
        if apply_sn
        else nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
    )


def global_pooling_2d(x):
    # global pooling means we take the average or sum of each feature map (dim=[2, 3])
    return torch.sum(x, dim=[2, 3], keepdim=True)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ConditionalBatchNorm2d, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.linear1 = nn.Linear(num_classes, in_channels)
        self.linear2 = nn.Linear(num_classes, in_channels)

    def forward(self, x, y):
        x = self.bn(x)
        gamma = self.linear1(y)[:, :, None, None]
        beta = self.linear2(y)[:, :, None, None]
        return x * gamma + beta


class Upsample(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(Upsample, self).__init__()

        self.conv = conv_2d(
            in_channels, out_channels, kernel_size, stride, padding, apply_sn=True
        )
        self.bn = ConditionalBatchNorm2d(num_classes, in_channels)
        self.act = nn.LeakyReLU(0.1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x, y):
        # start from bn and act because of the input order
        h = self.bn(x, y)
        h = self.act(h)
        h = self.up(h)
        h = self.conv(h)
        return h


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(Downsample, self).__init__()

        self.conv = conv_2d(
            in_channels, out_channels, kernel_size, stride, padding, apply_sn=True
        )
        self.act = nn.LeakyReLU(0.1)
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        h = self.conv(x)
        h = self.act(h)
        h = self.down(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.q_conv = conv_2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.k_conv = conv_2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.v_conv = conv_2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.o_conv = conv_2d(
            in_channels // 8,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # shape: [B, C, H, W] -> [B, H*W, C//8] (batch_size, seq_len, embed_dim)
        q = self.q_conv(x).view(B, C // 8, H * W).permute(0, 2, 1)
        k = self.k_conv(x).view(B, C // 8, H * W)
        v = self.v_conv(x).view(B, C // 8, H * W)

        attention = F.softmax(torch.matmul(q, k), dim=-1)

        out = torch.matmul(v, attention.permute(0, 2, 1)).view(B, C // 8, H, W)
        # shape: [B, C, H, W]
        out = self.o_conv(out) * self.gamma + x

        return out


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        base_channel,
        num_classes,
    ):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        # 1x1 -> 4x4
        self.input_linear = linear(z_dim, base_channel * 4 * 4, apply_sn=True)

        self.up_blocks = nn.ModuleList()
        for i in range(3):
            # 0: 4x4 -> 8x8
            # 1: 8x8 -> 16x16
            # 2: 16x16 -> 32x32
            self.up_blocks.append(
                Upsample(
                    num_classes,
                    in_channels=base_channel // (2**i),
                    out_channels=base_channel // (2 ** (i + 1)),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        # before output layer, apply self-attention
        self.attention = SelfAttention(base_channel // 8)
        # output layer, so out_channels=3
        # 32x32 -> 64x64
        self.output_conv = Upsample(
            num_classes,
            in_channels=base_channel // 8,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, y):
        h = self.input_linear(x).view(x.shape[0], -1, 4, 4)
        for up_block in self.up_blocks:
            h = up_block(h, y)
        h = self.attention(h)
        h = self.output_conv(h, y)
        return h.tanh()


# projection discriminator
class Discriminator(nn.Module):
    def __init__(
        self,
        base_channel,
        num_classes,
    ):
        super(Discriminator, self).__init__()

        # input layer, so in_channels=3
        # 64x64 -> 32x32
        self.input_conv = Downsample(
            in_channels=3,
            out_channels=base_channel // 8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        for i in range(3, 0, -1):
            # 3: 32x32 -> 16x16
            # 2: 16x16 -> 8x8
            # 1: 8x8 -> 4x4
            self.down_blocks.append(
                Downsample(
                    in_channels=base_channel // (2**i),
                    out_channels=base_channel // (2 ** (i - 1)),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        self.attention = SelfAttention(base_channel // 8)

        self.condition_linear = linear(num_classes, base_channel)
        self.output_linear = linear(base_channel, 1)
        self.aux_classifier = linear(base_channel, num_classes)

    def forward(self, x, y):
        h = self.input_conv(x)
        h = self.attention(h)
        for down_block in self.down_blocks:
            h = down_block(h)

        h = global_pooling_2d(h).view(x.shape[0], -1)

        condition = self.condition_linear(y)
        # project means do inner product between condition and h
        # (this is projection discriminator)
        projection = torch.sum(condition * h, dim=-1, keepdim=True)
        out = self.output_linear(h) + projection
        aux_out = self.aux_classifier(h)

        return out, aux_out


if __name__ == "__main__":
    z = torch.ones(4, 32)
    y = torch.ones(4, 24)
    generator = Generator(32, 256, 24)
    discriminator = Discriminator(256, 24)

    image = generator(z, y)
    out, aux_out = discriminator(image, y)

    print(image.shape)
    print(out.shape, aux_out.shape)
