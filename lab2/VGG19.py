from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # initialize weights with N(0, 1e-2)
        # nn.init.normal_(self.conv.weight, mean=0, std=1e-2)
        # # bias is initialized to 0
        # nn.init.constant_(self.conv.bias, 0)

        # add batch normalization to prevent gradient vanishing
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(VGG19, self).__init__()

        # input size: (224x224)

        # block 1 (image size: 112x112)
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # block 2 (image size: 56x56)
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # block 3 (image size: 28x28)
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # block 4 (image size: 14x14)
        self.block4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # block 5 (image size: 7x7)
        self.block5 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # mlp for classification
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x
