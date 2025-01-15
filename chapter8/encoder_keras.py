import tensorflow as tf
import keras
from keras import layers
from positional_encoding_keras import positional_encoding
from multi_head_attention_keras import MultiHeadAttention


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
    def __init__(self, n_layers, d_model, n_heads, ddf,
                 input_vocab_size, max_seq_len, drop_rate=0.1):
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


encoder = Encoder(n_layers=2, d_model=512, n_heads=8, ddf=1024, input_vocab_size=5000, max_seq_len=200)
encoder_output = encoder(tf.random.uniform((32, 100)), False, None)
print(encoder_output.shape)
