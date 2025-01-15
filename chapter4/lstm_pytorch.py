import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=28, hidden_size=128, batch_first=True),
            nn.LSTM(input_size=128, hidden_size=128),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.rnn(x.view(-1, 28, 28))
        return out


model = RNN()
print(model)
