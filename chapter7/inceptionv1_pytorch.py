import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


def conv3x3(in_channels, out_channels, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.ReLU()
    )


def conv5x5(in_channels, out_channels, padding=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=padding),
        nn.ReLU()
    )


class InceptionModule(nn.Module):
    def __init__(self, in_channels, c1, c3_1, c3, c5_1, c5, cp_1):
        super().__init__()
        self.conv1 = conv1x1(in_channels, c1)
        self.conv3_1 = conv1x1(in_channels, c3_1)
        self.conv3 = conv3x3(c3_1, c3, padding=1)
        self.conv5_1 = conv1x1(in_channels, c5_1)
        self.conv5 = conv5x5(c5_1, c5, padding=2)
        self.pool_conv = conv1x1(in_channels, cp_1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3_1 = self.conv3_1(x)
        conv3 = self.conv3(conv3_1)
        conv5_1 = self.conv5_1(x)
        conv5 = self.conv5(conv5_1)
        pool = self.pool(x)
        pool_conv = self.pool_conv(pool)
        out = torch.cat([conv1, conv3, conv5, pool_conv], dim=1)
        return out


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv2_2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 48, 64)
        self.inception4e = InceptionModule(512, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 1000)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.maxpool5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.softmax(self.fc(x))
        return y


model = Inception()
print(model)
