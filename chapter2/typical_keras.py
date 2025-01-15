from keras import layers, models


def cnn(image_batch):
    # The input is transformed into a 28x28 image, and the number of channels is 1 since it is a grayscale image.
    x_image = layers.Reshape((28, 28, 1))(image_batch)
    h_conv1 = layers.Conv2D(filters=16, kernel_size=5, padding='same', activation='relu')(x_image)
    h_pool1 = layers.AveragePooling2D(pool_size=(2, 2))(h_conv1)
    h_conv2 = layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(h_pool1)
    h_pool2 = layers.AveragePooling2D(pool_size=(2, 2))(h_conv2)
    # In order to feed the feature image into the fully connected layer, it needs to be spread out
    h_pool2_flat = layers.Flatten()(h_pool2)
    h_fc1 = layers.Dense(256, activation='relu')(h_pool2_flat)
    # Get the predicted probability for a total of 10 numbers from 0-9, note that the activation function here is SoftMax to achieve normalization.
    _y = layers.Dense(10, activation='softmax')(h_fc1)
    return _y


x = layers.Input(shape=(784,))
y_ = layers.Input(shape=(10,))
y = cnn(x)
model = models.Model(x, y)
print(model.summary())
