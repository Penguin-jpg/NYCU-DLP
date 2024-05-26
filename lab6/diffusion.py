import math
from dataset import PAD

import torch
import torch.nn as nn

# modified from:
# 1. https://learnopencv.com/denoising-diffusion-probabilistic-models/
# 2. https://github.com/lucidrains/denoising-diffusion-pytorch


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=128, theta=10000):
        super(SinusoidalPositionEmbeddings, self).__init__()

        self.dim = dim
        self.theta = theta

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super(AttentionBlock, self).__init__()

        self.channels = channels
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.channels, num_heads=4, batch_first=True
        )

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(
            1, 2
        )  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(
            B, self.channels, H, W
        )  # [B, C, H*W] --> [B, C, H, W]
        return x + h


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        dropout_rate=0.1,
        time_emb_dim=512,
        apply_attention=False,
    ):
        super(ResnetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.SiLU()

        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.linear1 = nn.Linear(
            in_features=time_emb_dim, out_features=self.out_channels
        )

        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.shortcut = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # group 1
        h = self.act(self.gn1(x))
        h = self.conv1(h)

        # group 2
        # add in timestep embedding
        h += self.linear1(self.act(t))[:, :, None, None]

        # group 3
        h = self.act(self.gn2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual and attention
        h = h + self.shortcut(x)
        h = self.attention(h)

        return h


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()

        self.downsample = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x, *args):
        return self.upsample(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_classes=24,
        time_input_dim=128,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 2, 4, 8),
        attention_resoultions=(8,),
        dropout_rate=0.1,
    ):
        super(UNet, self).__init__()

        time_emd_dim = time_input_dim * 4
        self.time_emb_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=time_input_dim, theta=10000),
            nn.Linear(in_features=time_input_dim, out_features=time_emd_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_emd_dim, out_features=time_emd_dim),
        )

        # +1 for padding index
        self.label_emb = nn.Embedding(num_classes + 1, time_emd_dim, padding_idx=PAD)

        self.first = nn.Conv2d(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels

        resolution = 1
        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dim=time_emd_dim,
                    apply_attention=(resolution in attention_resoultions),
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != num_resolutions - 1:
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)
                resolution *= 2

        # Bottleneck in between
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dim=time_emd_dim,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dim=time_emd_dim,
                    apply_attention=False,
                ),
            )
        )

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dim=time_emd_dim,
                    apply_attention=(resolution in attention_resoultions),
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))
                resolution //= 2

        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x, t, y):
        # combine timestep embedding and label emdding
        # emb = self.time_emb_mlp(t) + self.label_emb(y)
        time_emb = self.time_emb_mlp(t)
        # since it is a multi-label task, I will sum the embeddings for each label
        label_embed = self.label_emb(y).sum(dim=1)
        emb = time_emb + label_embed

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, emb)

        h = self.final(h)

        return h


# if __name__ == "__main__":
#     unet = UNet(
#         input_channels=3,
#         output_channels=3,
#         num_classes=3,
#         time_input_dim=128,
#         num_res_blocks=2,
#         base_channels=128,
#         base_channels_multiples=(1, 2, 4, 8),
#         attention_resoultions=(16, 8),
#         dropout_rate=0.1,
#     )

#     x = torch.randn([1, 3, 64, 64])
#     t = torch.ones([1], dtype=torch.long) * 100
#     y = torch.tensor([[0, 1, 1]], dtype=torch.long)
#     print(unet(x, t, y).shape)
