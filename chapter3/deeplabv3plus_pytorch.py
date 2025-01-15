import torch
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


class Deeplabv3Plus(nn.Module):
    def __init__(self):
        super(Deeplabv3Plus, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1_block = self._make_layer(64, 64, 3)
        self.conv2_block = self._make_layer(64, 128, 4, stride=2)
        self.conv3_block = self._make_layer(128, 256, 6, stride=2)
        self.conv4_block = self._make_layer(256, 512, 3, dilation=2)
        self.aspp = self._make_aspp(512, 256)
        self.aspp_b1 = self._aspp_layer(256, 6)
        self.aspp_b2 = self._aspp_layer(256, 12)
        self.aspp_b3 = self._aspp_layer(256, 18)
        self.aspp_b4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.aspp_b5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(64, 48, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(304, 256, kernel_size=3, bias=False)
        self.conv5 = nn.Conv2d(256, 1000, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1, dilation=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride=stride, downsample=(stride != 1 or dilation != 1), dilation=dilation))
        for i in range(1, n_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

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
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x_b1 = self.conv1_block(x)
        x_b2 = self.conv2_block(x_b1)
        x_b3 = self.conv3_block(x_b2)
        x_b4 = self.conv4_block(x_b3)
        print(x_b4.shape)
        x = self.aspp(x_b4)
        x1 = self.aspp_b1(x)
        x2 = self.aspp_b2(x)
        x3 = self.aspp_b3(x)
        x4 = self.aspp_b4(x)
        x5 = self.aspp_b5(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x_b1 = self.conv3(x_b1)
        # Blending up the underlying features through splicing.
        x = torch.cat([x, x_b1], dim=1)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.softmax(x)
        return x


deeplabv3plus = Deeplabv3Plus()
print(deeplabv3plus)
