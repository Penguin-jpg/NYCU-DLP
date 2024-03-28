from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(VGG19, self).__init__()

        # input size: (224x224)

        # block 1 (image size: 112x112)
        self.block1 = nn.Sequential(Block(in_channels, 64), nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2 (image size: 56x56)
        self.block2 = nn.Sequential(Block(64, 128), nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3 (image size: 28x28)
        self.block3 = nn.Sequential(Block(128, 256), Block(256, 256), nn.MaxPool2d(kernel_size=2, stride=2))

        # block 4 (image size: 14x14)
        self.block4 = nn.Sequential(Block(256, 512), Block(512, 512), nn.MaxPool2d(kernel_size=2, stride=2))

        # block 5 (image size: 7x7)
        self.block5 = nn.Sequential(Block(512, 512), Block(512, 512), nn.MaxPool2d(kernel_size=2, stride=2))

        # mlp for classification
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True)),
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
