import torch.optim as optim

# Stochastic gradient descent with momentum.
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # lr is the learning rate, and momentum is the weight of the momentum.

# Adam
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # lr is the learning rate, and betas are the decay rates for the first moment and the infinity norm.
