from keras import layers, models


def rnn(image_batch):
    # A total of 28 time points, each with a 28-dimensional vector of inputs.
    x_image = layers.Reshape((28, 28))(image_batch)
    # The return_sequences here must be set to True for the subsequent RNN layers to work properly.
    h_rnn1 = layers.SimpleRNN(units=128, return_sequences=True)(x_image)
    h_rnn2 = layers.SimpleRNN(units=128)(h_rnn1)
    _y = layers.Dense(10, activation='softmax')(h_rnn2)
    return _y


x = layers.Input(shape=(784,))
y_ = layers.Input(shape=(10,))
y = rnn(x)
model = models.Model(x, y)
print(model.summary())
