from keras import layers


# Create a neural network model with Dropout.
def fcnn(image_batch):
    h_fc1 = layers.Dense(200, input_dim=784)(image_batch)
    h_fc2 = layers.Dense(200)(h_fc1)
    h_dropout = layers.Dropout(0.5)(h_fc2)
    _y = layers.Dense(10, activation='softmax')(h_dropout)
    return _y
