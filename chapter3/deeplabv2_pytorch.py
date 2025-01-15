import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, dilation=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class Deeplabv2(nn.Module):
    def __init__(self):
        super(Deeplabv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1_block = self._make_layer(64, 64, 3)
        self.conv2_block = self._make_layer(64, 128, 4, stride=2)
        self.conv3_block = self._make_layer(128, 256, 6, stride=2)
        self.conv4_block = self._make_layer(256, 512, 3, dilation=2)
        self.aspp = self._make_aspp(512, 256)
        self.aspp_b1 = self._aspp_layer(256, 6)
        self.aspp_b2 = self._aspp_layer(256, 12)
        self.aspp_b3 = self._aspp_layer(256, 18)
        self.aspp_b4 = self._aspp_layer(256, 24)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1, dilation=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=(stride != 1 or dilation != 1), dilation=dilation))
        for i in range(1, n_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    # One of the cores of DeepLabv2 is the ASPP network.
    def _make_aspp(self, in_channels, out_channels):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
        return nn.Sequential(*modules)

    def _aspp_layer(self, channels, dilation):
        layers = []
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        layers.append(nn.Conv2d(channels, 1000, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        x = self.aspp(x)
        x1 = self.aspp_b1(x)
        x2 = self.aspp_b2(x)
        x3 = self.aspp_b3(x)
        x4 = self.aspp_b4(x)
        x = x1 + x2 + x3 + x4
        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)
        x = self.softmax(x)
        return x


deeplabv2 = Deeplabv2()
print(deeplabv2)
