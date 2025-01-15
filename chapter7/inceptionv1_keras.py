from functools import partial
from keras import layers, models

conv1x1 = partial(layers.Conv2D, kernel_size=1, activation='relu')
conv3x3 = partial(layers.Conv2D, kernel_size=3, padding='same', activation='relu')
conv5x5 = partial(layers.Conv2D, kernel_size=5, padding='same', activation='relu')


def inception_module(in_tensor, c1, c3_1, c3, c5_1, c5, cp_1):
    conv1 = conv1x1(c1)(in_tensor)
    conv3_1 = conv1x1(c3_1)(in_tensor)
    conv3 = conv3x3(c3)(conv3_1)
    conv5_1 = conv1x1(c5_1)(in_tensor)
    conv5 = conv5x5(c5)(conv5_1)
    pool = layers.MaxPool2D(3, strides=1, padding="same")(in_tensor)
    pool_conv = conv1x1(cp_1)(pool)
    merged = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool_conv])
    return merged


def inception(in_tensor):
    conv1 = layers.Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_tensor)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv1)
    conv2_1 = conv1x1(64)(pool1)
    conv2_2 = conv3x3(192)(conv2_1)
    pool2 = layers.MaxPool2D(3, 2, padding='same')(conv2_2)
    inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
    inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64)
    pool3 = layers.MaxPool2D(3, 2, padding='same')(inception3b)
    inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
    inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64)
    inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64)
    inception4d = inception_module(inception4c, 112, 144, 288, 32, 48, 64)
    inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128)
    pool4 = layers.MaxPool2D(3, 2, padding='same')(inception4e)
    inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
    inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)
    pool5 = layers.MaxPool2D(3, 2, padding='same')(inception5b)
    avg_pool = layers.GlobalAvgPool2D()(pool5)
    dropout = layers.Dropout(0.4)(avg_pool)
    _y = layers.Dense(1000, activation="softmax")(dropout)
    return _y


x = layers.Input(shape=(224, 224, 3))
y = inception(x)
model = models.Model(x, y)
print(model.summary())
