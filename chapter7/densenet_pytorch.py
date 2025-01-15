import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
            nn.Sequential(
            nn.BatchNorm2d(in_channels + i * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=1, stride=1, padding=1, bias=True,),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=True,)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(torch.cat(features, dim=1))
            features.append(x)
        return torch.cat(features, dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=True
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return 


class DenseNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        num_features = 64
        block_config = [6, 12, 24, 16]
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.softmax(self.fc(x))
        return y


model = DenseNet()
print(model)
