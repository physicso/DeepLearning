from keras import layers, models


def alexnet(image_batch):
    h_conv1 = layers.Conv2D(filters=96, kernel_size=11, strides=4,
                            padding='valid', activation='relu', use_bias=True)(image_batch)
    h_pool1 = layers.MaxPooling2D(pool_size=3, strides=2)(h_conv1)
    h_conv2 = layers.Conv2D(filters=256, kernel_size=5,
                            padding='same', activation='relu', use_bias=True)(h_pool1)
    h_pool2 = layers.MaxPooling2D(pool_size=3, strides=2)(h_conv2)
    h_conv3 = layers.Conv2D(filters=384, kernel_size=3,
                            padding='same', activation='relu', use_bias=True)(h_pool2)
    h_conv4 = layers.Conv2D(filters=384, kernel_size=3,
                            padding='same', activation='relu', use_bias=True)(h_conv3)
    h_conv5 = layers.Conv2D(filters=256, kernel_size=3,
                            padding='same', activation='relu', use_bias=True)(h_conv4)
    h_pool3 = layers.MaxPooling2D(pool_size=3, strides=2)(h_conv5)
    h_pool3_flat = layers.Flatten()(h_pool3)
    h_fc1 = layers.Dense(4096, activation='relu')(h_pool3_flat)
    h_fc2 = layers.Dense(4096, activation='relu')(h_fc1)
    _y = layers.Dense(1000, activation='softmax')(h_fc2)
    return _y


x = layers.Input(shape=(227, 227, 3))
y_ = layers.Input(shape=(1000,))
y = alexnet(x)
model = models.Model(x, y)
print(model.summary())
