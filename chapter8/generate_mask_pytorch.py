import torch


def generate_padding_mask(seq):
    seq = torch.tensor(seq)
    seq = torch.eq(seq, 0).float()
    return seq.unsqueeze(1).unsqueeze(1)


input_tensor = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
print(generate_padding_mask(input_tensor))


def generate_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)))
    return mask


mask = generate_look_ahead_mask(3)
print(mask)
