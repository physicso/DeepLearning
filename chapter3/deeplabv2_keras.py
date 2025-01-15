from keras import layers, models


def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)


def conv3(in_tensor, filters, dilation=1):
    conv = layers.Conv2D(filters, kernel_size=3, dilation_rate=dilation, padding='same')(in_tensor)
    return _after_conv(conv)


def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)


def conv1(in_tensor, filters, dilation=1):
    conv = layers.Conv2D(filters, kernel_size=1, dilation_rate=dilation, padding='same')(in_tensor)
    return _after_conv(conv)


def simple_conv1(in_tensor, filters, dilation=1):
    conv = layers.Conv2D(filters, kernel_size=1, dilation_rate=dilation, padding='same')(in_tensor)
    return conv


def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)


def resnet_block(in_tensor, filters, downsample=False, dilation=1):
    if downsample and dilation == 1:
        conv1_rb = conv3_downsample(in_tensor, filters)
    else:
        conv1_rb = conv3(in_tensor, filters, dilation)
    conv2_rb = conv3(conv1_rb, filters, dilation)
    if downsample and dilation == 1:
        in_tensor = conv1_downsample(in_tensor, filters)
    elif downsample:
        in_tensor = conv1(in_tensor, filters, dilation)
    result = layers.Add()([conv2_rb, in_tensor])
    return layers.Activation('relu')(result)


def block(in_tensor, filters, n_block, downsample=False, dilation=1):
    res = in_tensor
    for i in range(n_block):
        if i == 0:
            res = resnet_block(res, filters, downsample, dilation)
        else:
            res = resnet_block(res, filters, False, dilation)
    return res


# One of the centers of DeepLabv2 is the ASPP network
def aspp(in_tensor):
    dilation_rates = [1, 2, 3, 4] * 6
    conv_1 = simple_conv1(simple_conv1(conv3(in_tensor, 256, dilation_rates[0]), 256), 1000)
    conv_2 = simple_conv1(simple_conv1(conv3(in_tensor, 256, dilation_rates[1]), 256), 1000)
    conv_3 = simple_conv1(simple_conv1(conv3(in_tensor, 256, dilation_rates[2]), 256), 1000)
    conv_4 = simple_conv1(simple_conv1(conv3(in_tensor, 256, dilation_rates[3]), 256), 1000)
    result = layers.Add()([conv_1, conv_2, conv_3, conv_4])
    return result


def deeplabv2(image_batch):
    conv = layers.Conv2D(64, 7, strides=2, padding='same')(image_batch)
    conv = _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    # Reuse ResNet's block function.
    conv1_block = block(pool1, 64, 3, False)
    conv2_block = block(conv1_block, 128, 4, True)
    conv3_block = block(conv2_block, 256, 6, True)
    conv4_block = block(conv3_block, 512, 3, True, dilation=2)
    result = aspp(conv4_block)
    _y = layers.Activation('softmax')(layers.UpSampling2D(16)(result))
    return _y


x = layers.Input(shape=(224, 224, 3))
y = deeplabv2(x)
model = models.Model(x, y)
print(model.summary())
