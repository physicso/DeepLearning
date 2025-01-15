from keras import layers, models


def DenseLayer(x, nb_filter, bn_size=4):
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)
    return x


def DenseBlock(x, n_layers, growth_rate):
    for _ in range(n_layers):
        conv = DenseLayer(x, nb_filter=growth_rate)
        x = layers.concatenate([x, conv], axis=3)
    return x


def TransitionLayer(x, compression=0.5):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    return x


def densenet(input_image, growth_rate=32):
    conv1 = layers.Conv2D(growth_rate * 2, (7, 7), strides=2, padding='same')(input_image)
    conv1_bn = layers.BatchNormalization()(conv1)
    conv1_act = layers.ReLU()(conv1_bn)
    conv1_pool = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv1_act)
    dense1 = DenseBlock(conv1_pool, 6, growth_rate)
    trans1 = TransitionLayer(dense1)
    dense2 = DenseBlock(trans1, 12, growth_rate)
    trans2 = TransitionLayer(dense2)
    dense3 = DenseBlock(trans2, 24, growth_rate)
    trans3 = TransitionLayer(dense3)
    dense4 = DenseBlock(trans3, 16, growth_rate)
    feature = layers.GlobalAveragePooling2D()(dense4)
    _y = layers.Dense(1000, activation='softmax')(feature)
    return _y


x = models.Input(shape=(256, 256, 3))
y = densenet(x)
model = models.Model(x, y)
print(model.summary())
