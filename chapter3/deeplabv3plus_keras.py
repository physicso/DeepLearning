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


def aspp(in_tensor):
    dilation_rates = [1, 2, 3] * 6
    image_pooling = layers.AveragePooling2D(pool_size=(in_tensor.shape[-3], in_tensor.shape[-2]))(in_tensor)
    image_pooling = layers.Conv2D(256, 1, strides=1, padding='same', use_bias=False)(image_pooling)
    image_pooling = layers.BatchNormalization()(image_pooling)
    image_pooling = layers.Activation("relu")(image_pooling)
    image_pooling = layers.UpSampling2D([in_tensor.shape[-3], in_tensor.shape[-2]])(image_pooling)
    conv_1 = conv1(in_tensor, 256)
    conv_2 = conv3(in_tensor, 256, dilation_rates[0])
    conv_3 = conv3(in_tensor, 256, dilation_rates[1])
    conv_4 = conv3(in_tensor, 256, dilation_rates[2])
    result = layers.Concatenate()([image_pooling, conv_1, conv_2, conv_3, conv_4])
    return layers.UpSampling2D(4)(conv1(result, 256))


def deeplabv3plus(image_batch):
    conv = layers.Conv2D(64, 7, strides=2, padding='same')(image_batch)
    conv = _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    conv1_block = block(pool1, 64, 3, False)
    conv2_block = block(conv1_block, 128, 4, True)
    conv3_block = block(conv2_block, 256, 6, True)
    conv4_block = block(conv3_block, 512, 3, True, dilation=2)
    encode_result = aspp(conv4_block)
    # Blending up the underlying features through splicing
    concat_result = layers.Concatenate()([simple_conv1(conv1_block, 48), encode_result])
    result = conv3(concat_result, 256)
    _y = layers.Activation('softmax')(layers.UpSampling2D(4)(simple_conv1(result, 1000)))
    return _y


x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
y = deeplabv3plus(x)
model = models.Model(x, y)
print(model.summary())
