import tensorflow as tf
from keras import layers


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


mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 50, 512))
output, att = mha(y, y, y, mask=None)
print(output.shape, att.shape)
