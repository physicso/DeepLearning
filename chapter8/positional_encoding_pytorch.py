import torch


def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.float32(d_model))
    return pos * angle_rates


def positional_encoding(seq_len, d_model):
    positional_encoding = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    positional_encoding = positional_encoding.unsqueeze(0)
    return positional_encoding


pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)
