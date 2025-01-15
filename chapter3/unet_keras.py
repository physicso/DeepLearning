from keras import models, layers


# Construct a two-layer 3×3 convolution for easy subsequent reuse.
def conv(in_tensor, filters):
    _conv1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(in_tensor)
    _conv2 = layers.Conv2D(filters, 3, activation='relu', padding='same')(_conv1)
    return _conv2


# U-Net has a very organized structure.
def u_net(image_batch):
    conv1 = conv(image_batch, 32)
    pool1 = layers.MaxPooling2D(pool_size=2)(conv1)
    conv2 = conv(pool1, 64)
    pool2 = layers.MaxPooling2D(pool_size=2)(conv2)
    conv3 = conv(pool2, 128)
    pool3 = layers.MaxPooling2D(pool_size=2)(conv3)
    conv4 = conv(pool3, 256)
    pool4 = layers.MaxPooling2D(pool_size=2)(conv4)
    conv5 = conv(pool4, 512)
    up1 = layers.Concatenate(axis=3)([conv4, layers.Conv2DTranspose(256, 2, strides=2, padding='same')(conv5)])
    conv6 = conv(up1, 256)
    up2 = layers.Concatenate(axis=3)([conv3, layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv6)])
    conv7 = conv(up2, 128)
    up3 = layers.Concatenate(axis=3)([conv2, layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv7)])
    conv8 = conv(up3, 64)
    up4 = layers.Concatenate(axis=3)([conv1, layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv8)])
    conv9 = conv(up4, 32)
    # U-Net was first applied to binary classification
    _y = layers.Conv2D(2, 1, activation='sigmoid')(conv9)
    return _y


# Note that the U-Net input is 512x512.
x = models.Input(shape=(512, 512, 3))
y = u_net(x)
model = models.Model(x, y)
print(model.summary())
