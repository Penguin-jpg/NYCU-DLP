import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# modified from https://github.com/leo27945875/MHingeGAN-for-multi-label-conditional-generation/tree/master


def conv2d(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_sn=True
):
    return (
        spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
        if apply_sn
        else nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


class GlobalPooling2d(nn.Module):
    def __init__(self, type):
        super(GlobalPooling2d, self).__init__()
        self.type = type

    def forward(self, x):
        # global pooling means we take the average or sum of each feature map (dim=[2, 3])
        if self.type == "sum":
            return torch.sum(x, dim=[2, 3], keepdim=True)
        elif self.type == "avg":
            return torch.mean(x, dim=[2, 3], keepdim=True)
        else:
            raise NotImplementedError


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ConditionalBatchNorm2d, self).__init__()
        self.weight = nn.Linear(num_classes, num_channels)
        self.bias = nn.Linear(num_classes, num_channels)
        self.bn = nn.BatchNorm2d(num_channels, affine=False)
        nn.init.orthogonal_(self.weight.weight.data)
        nn.init.zeros_(self.bias.weight.data)

    def forward(self, input, c):
        output = self.bn(input)
        weight = self.weight(c).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias(c).unsqueeze(-1).unsqueeze(-1)
        return weight * output + bias


class Upsample(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_shortcut=False,
    ):
        super(Upsample, self).__init__()

        self.conv = conv2d(
            in_channels, out_channels, kernel_size, stride, padding, apply_sn=True
        )

        self.shortcut = (
            nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    apply_sn=True,
                ),
            )
            if use_shortcut
            else None
        )

        self.bn = ConditionalBatchNorm2d(num_classes, in_channels)
        self.act = nn.LeakyReLU(0.1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x, c):
        h = self.bn(x, c)
        h = self.act(h)
        h = self.up(h)
        h = self.conv(h)
        return h + self.shortcut(x) if self.shortcut is not None else h


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_shortcut=False,
    ):
        super(Downsample, self).__init__()

        self.conv = conv2d(
            in_channels, out_channels, kernel_size, stride, padding, apply_sn=True
        )

        self.shortcut = (
            nn.Sequential(
                nn.AvgPool2d(2),
                conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    apply_sn=True,
                ),
            )
            if use_shortcut
            else None
        )

        self.act = nn.LeakyReLU(0.1)
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        h = self.conv(x)
        h = self.act(h)
        h = self.down(h)
        return h + self.shortcut(x) if self.shortcut else h


class BatchStd(nn.Module):
    def __init__(self):
        super(BatchStd, self).__init__()

    def forward(self, x):
        B, H, W = x.size(0), x.size(2), x.size(3)
        std = torch.std(x, dim=0).mean().repeat(B, 1, H, W)
        return torch.cat([x, std], dim=1)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.q_proj = conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.k_proj = conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.v_proj = conv2d(
            in_channels,
            in_channels // 8,
            kernel_size=1,
            stride=1,
            padding=0,
            apply_sn=True,
        )
        self.o_proj = conv2d(
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
        q = self.q_proj(x).view(B, C // 8, H * W).permute(0, 2, 1)
        k = self.k_proj(x).view(B, C // 8, H * W)
        v = self.v_proj(x).view(B, C // 8, H * W)

        attention = F.softmax(q @ k, dim=-1)

        out = (v @ attention.permute(0, 2, 1)).view(B, C // 8, H, W)
        # shape: [B, C, H, W]
        out = self.o_proj(out) * self.gamma + x

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
        self.linear = spectral_norm(nn.Linear(z_dim, base_channel * 4 * 4))

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

    def forward(self, x, c):
        h = self.linear(x).view(x.size(0), -1, 4, 4)
        # print(f"first h: {h.shape}")
        for up_block in self.up_blocks:
            h = up_block(h, c)
            # print(f"up_block: {h.shape}")
        h = self.attention(h)
        h = self.output_conv(h, c)
        image = torch.tanh(h)
        return image


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
        self.global_pool = nn.Sequential(
            BatchStd(),
            Downsample(
                in_channels=base_channel + 1,
                out_channels=base_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            GlobalPooling2d("sum"),
            nn.Flatten(),
        )

        self.condition_linear = spectral_norm(nn.Linear(num_classes, base_channel))
        self.output_linear = spectral_norm(nn.Linear(base_channel, 1))

        self.aux = spectral_norm(nn.Linear(base_channel, num_classes))

    def forward(self, x, c):
        h = self.input_conv(x)
        h = self.attention(h)
        for down_block in self.down_blocks:
            h = down_block(h)
        h = self.global_pool(h)

        condition = self.condition_linear(c)
        # project means do inner product between condition and h
        projection = torch.sum(condition * h, dim=-1, keepdim=True)
        out = self.output_linear(h) + projection
        aux_out = self.aux(h)

        return out, aux_out


# if __name__ == "__main__":

#     c = torch.ones(4, 24)
#     x = torch.ones(4, 32)
#     g = Generator(32, 256, 24)
#     d = Discriminator(256, 24)

#     image = g(x, c)
#     out, aux_out = d(image, c)

#     print(image.shape)
#     print(out.shape, aux_out.shape)
