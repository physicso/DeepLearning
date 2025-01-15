from keras import optimizers

# Stochastic gradient descent with momentum.
sgd = optimizers.SGD(lr=0.01, momentum=0.9)  # `lr` is the learning rate, and `momentum` is the momentum weight.

# Adam
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)  # lr is the learning rate, and beta_1 and beta_2 are the decay rates for the first moment and the infinity norm, respectively.
