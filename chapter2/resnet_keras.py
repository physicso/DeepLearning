from keras import layers, models


# Abstract a common function that imposes a batch normalization and activation function after the convolutional layer.
def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)


# Because of the need for batch normalization, use_bias=False is set in all later convolutional layer definitions.
# Ordinary 3x3 convolutional layers
def conv3(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


# Stride is a 3x3 convolutional layer of 2.
def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


# Ordinary 1x1 convolutional layers.
def conv1(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


# Stride is a 1x1 convolutional layer of 2.
def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same', use_bias=False)(in_tensor)
    return _after_conv(conv)


# Definition of residual units.
def resnet_block(in_tensor, filters, downsample=False):
    # ResNet has the problem of dimension matching, which is solved here with the downsample metrics.
    if downsample:
        h_conv1 = conv3_downsample(in_tensor, filters)
    else:
        h_conv1 = conv3(in_tensor, filters)
    h_conv2 = conv3(h_conv1, filters)

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    result = layers.Add()([h_conv2, in_tensor])

    return layers.Activation('relu')(result)


# Implementation blocks are defined with downsample=True for all blocks except Block 1.
def block(in_tensor, filters, n_block, downsample=False, convx=resnet_block):
    res = in_tensor
    for index in range(n_block):
        if index == 0:
            res = convx(res, filters, downsample)
        else:
            res = convx(res, filters, False)
    return res


def resnet(image_batch, convx=resnet_block):
    conv = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(image_batch)
    conv = _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    conv1_block = block(pool1, 64, 3, False, convx)
    conv2_block = block(conv1_block, 128, 4, True, convx)
    conv3_block = block(conv2_block, 256, 6, True, convx)
    conv4_block = block(conv3_block, 512, 3, True, convx)
    # Using GAP to solve the problem of excessive amount of parameters in the fully connected layer.
    pool2 = layers.GlobalAvgPool2D()(conv4_block)
    _y = layers.Dense(1000, activation='softmax')(pool2)
    return _y


x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
y = resnet(x)
model = models.Model(x, y)
print(model.summary())
