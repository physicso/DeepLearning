from keras import layers, models


def xception_block_A(x, filters, r = True):
    residual = layers.Conv2D(filters[0], (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    if r:
        x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters[1], (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(filters[2], (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    return x


def xception_block_B(x, repeat, filters):
    residual = x
    for index in range(repeat):
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters[index], (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    return x


def xception(img_input, classes=1000):
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = xception_block_A(x, [128, 128, 128], r=False)
    x = xception_block_A(x, [256, 256, 256])
    x = xception_block_A(x, [728, 728, 728])
    for i in range(8):
        x = xception_block_B(x, 3, [728, 728, 728])
    x = xception_block_A(x, [1024, 728, 1024])
    x = layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    _y = layers.Dense(classes, activation='softmax')(x)
    return _y


x = layers.Input(shape=(224, 224, 3))
y = xception(x)
model = models.Model(x, y)
print(model.summary())
