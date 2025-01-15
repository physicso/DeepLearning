from keras import layers, models


def grouped_convolution_block(init, grouped_channels, cardinality, strides):
    channel_axis = -1
    group_list = []
    for c in range(cardinality):
        x = layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(init)
        x = layers.Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides))(x)
        group_list.append(x)
    group_merge = layers.concatenate(group_list, axis=channel_axis)
    x = layers.BatchNormalization()(group_merge)
    x = layers.Activation('relu')(x)
    return x


def block_module(x, filters, cardinality, strides):
    init = x
    grouped_channels = int(filters / cardinality)
    if init.shape[-1] != 2 * filters:
        init = layers.Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides), use_bias=False)(init)
        init = layers.BatchNormalization()(init)
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = grouped_convolution_block(x, grouped_channels, cardinality, strides)
    x = layers.Conv2D(filters * 2, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([init, x])
    x = layers.Activation('relu')(x)
    return x


def resnext(img_input):
    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), use_bias=False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(x)

    for _ in range(3):
        x = block_module(x, 128, 8, 1)
    x = block_module(x, 256, 8, 2)
    for _ in range(2):
        x = block_module(x, 256, 8, 1)
    x = block_module(x, 512, 8, 2)
    for _ in range(2):
        x = block_module(x, 512, 8, 1)
    x = layers.GlobalAveragePooling2D()(x)
    _y = layers.Dense(1000, activation='softmax')(x)
    return _y


x = layers.Input(shape=(256, 256, 3))
y = resnext(x)
model = models.Model(x, y)
print(model.summary())
