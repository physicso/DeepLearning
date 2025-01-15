from keras import layers

regularizer_ratio = 0.1


# L1 regularization.
def fcnn(image_batch):
    h_fc1 = layers.Dense(200, input_dim=784,
                         kernel_regularizer=l1(regularizer_ratio) if regularizer_ratio != 0.0 else None)(image_batch)
    h_fc2 = layers.Dense(200, kernel_regularizer=l1(regularizer_ratio) if regularizer_ratio != 0.0 else None)(h_fc1)
    h_dropout = layers.Dropout(0.5)(h_fc2)
    _y = layers.Dense(10, activation='softmax')(h_dropout)
    return _y


# L2 regularization.
def fcnn(image_batch):
    h_fc1 = layers.Dense(200, input_dim=784,
                         kernel_regularizer=l2(regularizer_ratio) if regularizer_ratio != 0.0 else None)(image_batch)
    h_fc2 = layers.Dense(200, kernel_regularizer=l2(regularizer_ratio) if regularizer_ratio != 0.0 else None)(h_fc1)
    h_dropout = layers.Dropout(0.5)(h_fc2)
    _y = layers.Dense(10, activation='softmax')(h_dropout)
    return _y
