import tensorflow as tf
from keras import layers
from positional_encoding_keras import positional_encoding
from multi_head_attention_keras import MultiHeadAttention
from encoder_keras import feed_forward_network, Encoder


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
        h *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        h += self.pos_embedding[:, :seq_len, :]
        h = self.dropout(h, training=training)

        for i in range(self.n_layers):
            h = self.decoder_layers[i](h, encoder_out, training, look_ahead_mask, padding_mask)
        return h


encoder = Encoder(n_layers=2, d_model=512, n_heads=8, ddf=1024, input_vocab_size=5000, max_seq_len=200)
encoder_output = encoder(tf.random.uniform((32, 100)), False, None)
decoder = Decoder(n_layers=2, d_model=512, n_heads=8, ddf=1024, target_vocab_size=5000, max_seq_len=200)
decoder_output = decoder(tf.random.uniform((32, 100)), encoder_output, False, None, None)
print(decoder_output.shape)
