import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.block1 = self._make_block(3, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 3)
        self.block4 = self._make_block(256, 512, 3)
        self.block5 = self._make_block(512, 512, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.softmax(self.fc3(x))
        return y

    def _make_block(self, in_channels, out_channels, n_conv):
        conv_block = []
        for _ in range(n_conv):
            conv_block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            conv_block.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*conv_block)


model = VGGNet()
print(model)
