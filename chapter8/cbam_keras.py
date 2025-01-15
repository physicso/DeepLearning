import tensorflow as tf
from keras import layers, models


class CBAMLayer(layers.Layer):
    def __init__(self, channel):
        super(CBAMLayer, self).__init__()

        self.channel = channel
        self.max_pool = layers.GlobalMaxPooling2D()
        self.ave_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(channel // 8, activation=tf.nn.relu)
        self.dense2 = layers.Dense(channel)
        self.max = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))
        self.mean = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))
        self.conv = layers.Conv2D(1, kernel_size=(7, 7), strides=1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-4)

    def _mlp(self, x):
        x = tf.reshape(x, shape=(-1, 1, 1, self.channel))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def _channel_attention(self, x):
        # Global maximum pooling.
        x_max = self.max_pool(x)
        x_max = self._mlp(x_max)

        # Global average pooling.
        x_ave = self.ave_pool(x)
        x_ave = self._mlp(x_ave)

        x_sum = x_max + x_ave
        x = tf.multiply(x, tf.nn.sigmoid(x_sum))

        return x

    def _spatial_attention(self, x, training):
        # Maximum feature on the channel.
        x1 = self.max(x)
        # Average features on the channel.
        x2 = self.mean(x)
        x_sum = tf.concat([x1, x2], 3)

        x_sum = self.conv(x_sum)
        x_sum = self.bn(x_sum, training)

        x = tf.multiply(x, tf.nn.sigmoid(x_sum))

        return x

    def call(self, x, training=False):
        x = self._channel_attention(x)
        x = self._spatial_attention(x, training)
        return x


if __name__ == "__main__":
    channel = 512
    input = layers.Input(shape=(10, 10, 512))
    output = CBAMLayer(channel)(input)
    model = models.Model(input, output)
    model.summary()
