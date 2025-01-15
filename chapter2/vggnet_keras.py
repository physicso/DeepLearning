from functools import partial
from keras import layers, models

# Use partial to construct a simple convolutional layer with a fixed number of input parameters, and pass in only the number of convolutional kernels.
simple_conv2d = partial(layers.Conv2D,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu')


# VGGNet has five blocks, each of which is relatively similar, with the same number of convolution kernels in each of the internal convolutional layers.
def block(in_tensor, filters, n_conv):
    conv_block = in_tensor
    # Build multiple convolutional layers via for loops.
    for _ in range(n_conv):
        conv_block = simple_conv2d(filters=filters)(conv_block)
    return layers.MaxPooling2D()(conv_block)


def vggnet(image_batch):
    # After the above abstraction, only the number of convolutional kernels and the number of convolutional layers need to be passed in to construct VGGNet.
    # Here is an example for VGG-16, VGG-19 only needs to modify the number of convolutional layers in each block.
    block1 = block(image_batch, 64, 2)
    block2 = block(block1, 128, 2)
    block3 = block(block2, 256, 3)
    block4 = block(block3, 512, 3)
    block5 = block(block4, 512, 3)
    # VGGNet uses the spreading operation, which here generates a large number of parameters.
    flat = layers.Flatten()(block5)
    h_fc1 = layers.Dense(4096, activation='relu')(flat)
    h_fc2 = layers.Dense(4096, activation='relu')(h_fc1)
    _y = layers.Dense(1000, activation='softmax')(h_fc2)
    return _y


x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
y = vggnet(x)
model = models.Model(x, y)
print(model.summary())
