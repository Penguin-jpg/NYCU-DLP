from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, multiplier=4):
        super(Bottleneck, self).__init__()

        # conv(k=1, s=1, p=0) -> bn -> relu -> conv(k=3, s=1 or 2, p=1)
        # -> bn -> relu -> conv(k=1, s=1, p=0) -> bn -> residual -> relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        # for resnet50, the multiplier is 4
        self.conv3 = nn.Conv2d(out_channels, out_channels * multiplier, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * multiplier)

        # because the number of channels maybe different for in_channels and
        # out_channels, the shortcut needs to be a 1x1 conv to match the number of channels
        if in_channels != out_channels * multiplier or stride == 2:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels * multiplier,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()
        self.bn4 = nn.BatchNorm2d(out_channels * multiplier)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        x = self.shortcut(x)
        x = self.bn4(x)
        h += x
        h = self.relu(h)
        return h


class ResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Bottleneck(64, 64, stride=1),
            ),
            # use 64 as out_channels because it will be multiplied by 4 during
            # construction of the bottleneck
            Bottleneck(256, 64, stride=1),
            Bottleneck(256, 64, stride=1),
        )

        self.conv3 = nn.Sequential(
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1),
            Bottleneck(512, 128, stride=1),
        )

        self.conv4 = nn.Sequential(
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
            Bottleneck(1024, 256, stride=1),
        )

        self.conv5 = nn.Sequential(
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512, stride=1),
            Bottleneck(2048, 512, stride=1),
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
