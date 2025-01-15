from keras import layers, models


def lenet(image_batch):
    # The input image for LeNet is 32x32.
    x_image = layers.Reshape((32, 32, 1))(image_batch)
    # Unlike typical convolutional neural networks, the Padding of the convolutional layer in LeNet is Valid and the activation function is sigmoid.
    h_conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='valid',
                            activation='sigmoid')(x_image)
    # Average pooling is used here.
    h_pool1 = layers.AveragePooling2D(pool_size=2, padding='same')(h_conv1)
    h_conv2 = layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation='sigmoid')(h_pool1)
    h_pool2 = layers.AveragePooling2D(pool_size=2, padding='same')(h_conv2)
    h_pool2_flat = layers.Flatten()(h_pool2)
    h_fc1 = layers.Dense(120, activation='sigmoid')(h_pool2_flat)
    h_fc2 = layers.Dense(84, activation='sigmoid')(h_fc1)
    _y = layers.Dense(10, activation='softmax')(h_fc2)
    return _y


x = layers.Input(shape=(1024,))
y_ = layers.Input(shape=(10,))
y = lenet(x)
model = models.Model(x, y)
print(model.summary())
