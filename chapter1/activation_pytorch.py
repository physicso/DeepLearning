import torch.nn as nn

# Sigmoid
sigmoid = nn.Sigmoid()

# Tanh
tanh = nn.Tanh()

# ReLU
relu = nn.ReLU()

# LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.3)

# PReLU
prelu = nn.PReLU(num_parameters=1, init=0.0)  # num_parameters represents the number of parameters for the gradient when the input is less than 0, and init indicates the initial value.
