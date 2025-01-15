import torch
import torch.nn as nn


class CBAMLayer(nn.Module):
    def __init__(self, channels):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU()
        )
        self.dense2 = nn.Linear(channels // 8, channels)
        self.max = torch.max
        self.mean = torch.mean
        self.conv = nn.Conv2d(2, 1, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-4)

    def _mlp(self, x):
        batch_size, channels, _, _ = x.size()
        x = x.view(batch_size, channels)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def _channel_attention(self, x):
        # Global maximum pooling.
        x_max = self.max_pool(x)
        x_max = self._mlp(x_max)

        # Global average pooling.
        x_ave = self.ave_pool(x)
        x_ave = self._mlp(x_ave)

        x_sum = x_max + x_ave
        x = x * torch.sigmoid(x_sum.unsqueeze(2).unsqueeze(3))

        return x

    def _spatial_attention(self, x, training):
        # Maximum feature on the channel.
        x1, _ = self.max(x, dim=1, keepdim=True)
        # Average features on the channel.
        x2 = self.mean(x, dim=1, keepdim=True)
        x_sum = torch.cat([x1, x2], 1)

        x_sum = self.conv(x_sum)
        x_sum = self.bn(x_sum)

        x = x * torch.sigmoid(x_sum)

        return x

    def forward(self, x, training=False):
        x = self._channel_attention(x)
        x = self._spatial_attention(x, training)
        return x


if __name__ == "__main__":
    channels = 512
    input = torch.randn(1, channels, 10, 10)
    cbam_layer = CBAMLayer(channels)
    output = cbam_layer(input)
    print(output.shape)
