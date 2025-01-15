import torch
import torch.nn as nn

# Define important parameters.
n_hidden = 64
n_classes = 2
n_layers = 4
batch_size = 512
max_length = 50
frame_size = 300


# Obtain the true (non-zero) length of the time series data.
def length(seq):
    used = torch.sign(torch.max(torch.abs(seq), dim=2)[0])
    leng = torch.sum(used, dim=1)
    leng = leng.int()
    return leng


# Obtain the characterization vector corresponding to the actual length.
def last_relevant(output, length):
    index = torch.arange(0, batch_size) * max_length + (length - 1)
    flat = output.reshape([-1, n_hidden])
    relevant = flat[index]
    return relevant


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.layers = nn.ModuleList([nn.LSTM(frame_size, n_hidden, batch_first=True) for _ in range(n_layers)])
        self.dense = nn.Linear(n_hidden, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence):
        seq_length = length(sequence)
        layer = sequence
        for lstm in self.layers:
            layer, _ = lstm(layer)
        last = last_relevant(layer, seq_length)
        pred = self.dense(last)
        pred = self.softmax(pred)
        return pred


model = RNN()
print(model)
