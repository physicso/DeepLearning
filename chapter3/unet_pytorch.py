import torch
import torch.nn as nn


# Construct a two-layer 3x3 convolution for easy subsequent reuse.
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


# U-Net has a very organized structure.
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = Conv(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = Conv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = Conv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = Conv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv6 = Conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv7 = Conv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv8 = Conv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.conv9 = Conv(64, 32)
        self.final_conv = nn.Conv2d(32, 2, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        up1 = torch.cat([conv4, self.up1(conv5)], dim=1)
        conv6 = self.conv6(up1)
        up2 = torch.cat([conv3, self.up2(conv6)], dim=1)
        conv7 = self.conv7(up2)
        up3 = torch.cat([conv2, self.up3(conv7)], dim=1)
        conv8 = self.conv8(up3)
        up4 = torch.cat([conv1, self.up4(conv8)], dim=1)
        conv9 = self.conv9(up4)
        final_conv = self.final_conv(conv9)
        output = self.activation(final_conv)
        return output


u_net = UNet()
print(u_net)
