import numpy as np
import tensorflow as tf
import keras
from keras import layers


# The CNN feature extractor of TransUNet is replaced by the following simple CNN.
class SimpleCNN(layers.Layer):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cnn_2x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation='relu')
        self.cnn_4x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation='relu')
        self.cnn_8x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation='relu')
        self.cnn_16x = layers.Conv2D(512, kernel_size=3, strides=2, padding="same", activation='relu')

    def call(self, x):
        x_2x = self.cnn_2x(x)
        x_4x = self.cnn_4x(x_2x)
        x_8x = self.cnn_8x(x_4x)
        x_16x = self.cnn_16x(x_8x)
        return x_2x, x_4x, x_8x, x_16x


cnn = SimpleCNN()
x_2x, x_4x, x_8x, x_16x = cnn(tf.random.uniform((10, 224, 224, 3)))
print(x_2x.shape, x_4x.shape, x_8x.shape, x_16x.shape)


# Converts the last layer of features from the CNN feature extractor into a format that can be input into Transformer.
class AddPosEmbed(layers.Layer):
    def __init__(self):
        super(AddPosEmbed, self).__init__()

    def build(self, input_shape):
        self.pe = tf.Variable(name="pos_embed", initial_value=keras.initializers.RandomNormal()(
            shape=(1, input_shape[1], input_shape[2])), trainable=True, dtype=tf.float32)

    def call(self, x):
        return x + tf.cast(self.pe, dtype=x.dtype)


class TransEmbed(tf.keras.layers.Layer):
    def __init__(self, embed_dim=768, patch_size=1):
        super(TransEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.cnn = layers.Conv2D(filters=self.embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")
        self.pos_embed = AddPosEmbed()

    def call(self, x):
        B, H, W, C = x.shape
        x = self.cnn(x)
        x = tf.reshape(x, (B, H * W, self.embed_dim))
        x = self.pos_embed(x)
        return x


emb = TransEmbed()
out = emb(tf.random.uniform((10, 14, 14, 256)))
print(out.shape)


# The Transformer structure uses the features extracted from the 16x CNN as input.
class MutilHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads=8, use_bias=False, dropout=0.1):
        super(MutilHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.scale = self.depth ** -0.5

        self.qkv = layers.Dense(d_model * 3, use_bias=use_bias)
        self.attn_drop = layers.Dropout(dropout)

        self.proj = layers.Dense(d_model)
        self.proj_drop = layers.Dropout(dropout)

    def call(self, inputs, training):
        B, N, C = inputs.shape

        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.depth])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x


class MLP(layers.Layer):
    def __init__(self, d_model, diff, drop=0.1):
        super(MLP, self).__init__()
        self.fc1 = layers.Dense(diff, activation="gelu")
        self.drop = layers.Dropout(drop)
        self.fc2 = layers.Dense(d_model)

    def call(self, inputs, training):
        x = self.fc1(inputs)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        return x


class TransformerBlock(layers.Layer):

    def __init__(self, num_heads=8, mlp_dim=3072, dropout=0.1, d_model=768, use_bias=False):
        super(TransformerBlock, self).__init__()

        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = MutilHeadAttention(d_model, num_heads=num_heads, use_bias=use_bias, dropout=dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(d_model, d_model * 4, drop=dropout)

    def call(self, inputs, training):
        x = self.norm1(inputs)
        x = self.attn(x, training=training)
        x += inputs
        x2 = self.norm2(x)
        x2 = self.mlp(x2, training=training)
        x += x2
        return x


encoder = TransformerBlock()
out = encoder(tf.random.uniform((10, 196, 768)))
print(out.shape)


# The feature fusion part on the right side of TransUNet.
class DecoderBlock(layers.Layer):
    def __init__(self, filters):
        super(DecoderBlock, self).__init__()
        self.filters = filters

        self.conv1 = layers.Conv2D(filters=self.filters, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters=self.filters, kernel_size=3, padding='same')
        self.upsampling = layers.Conv2DTranspose(self.filters, 2, strides=2, padding='same')

    def call(self, inputs, skip=None):
        x = self.upsampling(inputs)
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, decoder_channels=[256, 128, 64, 16], n_skip=3):
        super(Decoder, self).__init__()
        self.decoder_channels = decoder_channels
        self.n_skip = n_skip

        self.decoders = [DecoderBlock(filters=d_model) for d_model in self.decoder_channels]

    def call(self, x, features):
        for i, decoder_block in enumerate(self.decoders):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


encoder = Decoder()
out = encoder(x=tf.random.uniform((10, 14, 14, 768)), features=[x_8x, x_4x, x_2x])
print(out.shape)


# A simple segmentation module is added to classify each pixel into 3 categories.
class Segmentation(layers.Layer):
    def __init__(self, num_classes=3, kernel_size=1):
        super(Segmentation, self).__init__()

        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(filters=self.num_classes, kernel_size=self.kernel_size, padding="same")

    def call(self, inputs):
        x = self.conv(inputs)
        return x


seg = Segmentation()
out = seg(tf.random.uniform((10, 224, 224, 16)))
print(out.shape)


# The final TransUNet model.
class TransUNet(keras.Model):
    def __init__(self, d_model=768, n_layers=2, num_heads=8, use_bias=False, mlp_dim=3072, dropout=0.1,
                 decoder_channels=[256, 128, 64, 16], n_skip=3, num_classes=3):
        super(TransUNet, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.cnn = SimpleCNN()
        self.emb = TransEmbed()

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.encoders = [
            TransformerBlock(num_heads=num_heads, d_model=d_model, use_bias=use_bias, mlp_dim=mlp_dim, dropout=dropout)
            for i in range(self.n_layers)]

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.conv = layers.Conv2D(filters=decoder_channels[0] * 2, kernel_size=3, padding="same")

        self.decoder = Decoder(decoder_channels=decoder_channels, n_skip=n_skip)
        self.seg_out = Segmentation(num_classes=num_classes)

    def call(self, inputs, training):
        feat_2x, feat_4x, feat_8x, feat_16x = self.cnn(inputs)

        trans_inp = self.emb(feat_16x)
        trans_inp = self.norm1(trans_inp)
        for encoder in self.encoders:
            trans_inp = encoder(trans_inp, training=training)
        trans_out = self.norm2(trans_inp)

        size = int(np.sqrt(trans_out.shape[1]))
        feat_16x = tf.reshape(trans_out, (-1, size, size, self.d_model))
        feat_16x = self.conv(feat_16x)

        feat = self.decoder(x=feat_16x, features=[feat_8x, feat_4x, feat_2x])
        logit = self.seg_out(feat)

        return logit


transunet = TransUNet()
out = transunet(tf.random.uniform((10, 224, 224, 3)))
print(out.shape)
