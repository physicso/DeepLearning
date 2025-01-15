from keras import layers, models


def rnn(image_batch):
    x_image = layers.Reshape((28, 28))(image_batch)
    h_rnn1 = layers.LSTM(units=128, return_sequences=True)(x_image)
    h_rnn2 = layers.LSTM(units=128)(h_rnn1)
    _y = layers.Dense(10, activation='softmax')(h_rnn2)
    return _y


x = layers.Input(shape=(784,))
y_ = layers.Input(shape=(10,))
y = rnn(x)
model = models.Model(x, y)
print(model.summary())
