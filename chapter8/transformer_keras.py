import numpy as np
import tensorflow as tf
import keras
from keras import layers


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]


def generate_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q_len = tf.shape(q)[1]
        kv_len = tf.shape(k)[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = tf.reshape(q, (batch_size, q_len, self.num_heads, self.depth))
        k = tf.reshape(k, (batch_size, kv_len, self.num_heads, self.depth))
        v = tf.reshape(v, (batch_size, kv_len, self.num_heads, self.depth))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        output = tf.transpose(output, [0, 2, 1, 3])

        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output, attention_weights


def feed_forward_network(d_model, diff):
    return keras.Sequential([
        layers.Dense(diff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = feed_forward_network(d_model, ddf)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf, input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = layers.Embedding(input_vocab_size, d_model, embeddings_initializer='normal')
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encode_layer = [EncoderLayer(d_model, n_heads, ddf, drop_rate) for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training, mask):
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb, training=training)

        for i in range(self.n_layers):
            x = self.encode_layer[i](x, training, mask)
        return x


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, inputs, encode_out, training, look_ahead_mask, padding_mask):
        att1, _ = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(inputs + att1)

        att2, _ = self.mha2(out1, encode_out, encode_out, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(out1 + att2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(out2 + ffn_out)

        return out3


class Decoder(layers.Layer):
    def __init__(self, n_layers, d_model, n_heads, ddf, target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model, embeddings_initializer='normal')
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decoder_layers = [DecoderLayer(d_model, n_heads, ddf, drop_rate) for _ in range(n_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(inputs)[1]
        h = self.embedding(inputs)
        h += self.pos_embedding[:, :seq_len, :]
        h = self.dropout(h, training=training)

        for i in range(self.n_layers):
            h = self.decoder_layers[i](h, encoder_out, training, look_ahead_mask, padding_mask)
        return h


class Transformer(keras.Model):
    def __init__(self, n_layers, d_model, n_heads, diff, input_vocab_size, target_vocab_size, max_seq_len,
                 drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, d_model, n_heads, diff, input_vocab_size, max_seq_len, drop_rate)

        self.decoder = Decoder(n_layers, d_model, n_heads, diff, target_vocab_size, max_seq_len, drop_rate)

        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, encode_padding_mask, look_ahead_mask, decode_padding_mask):
        encode_out = self.encoder(inputs, training, encode_padding_mask)
        decode_out = self.decoder(targets, encode_out, training, look_ahead_mask, decode_padding_mask)
        out = self.final_layer(decode_out)

        return out


transformer = Transformer(n_layers=2, d_model=512, n_heads=8, diff=1024, input_vocab_size=5000, target_vocab_size=6000,
                          max_seq_len=200)
inp = tf.random.uniform((32, 100))
target = tf.random.uniform((32, 200))
out = transformer(inp, target, training=False, encode_padding_mask=None, look_ahead_mask=None, decode_padding_mask=None)
print(out.shape)
