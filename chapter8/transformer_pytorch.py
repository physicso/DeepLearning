import torch
import torch.nn as nn
import torch.nn.functional as F

training = False


def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.float32(d_model))
    return pos * angle_rates


def positional_encoding(seq_len, d_model):
    positional_encoding = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    positional_encoding = positional_encoding.unsqueeze(0)
    return positional_encoding


def generate_padding_mask(seq):
    seq = seq.type(torch.float32)
    return seq[:, None, None, :]


def generate_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)))
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = torch.cast(torch.shape(k)[-1], torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.depth)

        output = self.fc(output)

        return output, attention


def feed_forward_network(d_model, diff):
    return nn.Sequential(nn.Linear(d_model, diff), nn.ReLU(), nn.Linear(diff, d_model))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = feed_forward_network(d_model, ddf)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs, mask):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output)
        out1 = self.layernorm1(inputs + att_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, ddf, input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.encode_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ddf, drop_rate) for _ in range(n_layers)])

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs, mask):
        seq_len = inputs.size(1)
        word_emb = self.embedding(inputs)
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb)

        for i in range(self.n_layers):
            x = self.encode_layers[i](x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ddf, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, n_heads)
        self.mha2 = MultiHeadAttention(d_model, n_heads)
        self.ffn = feed_forward_network(d_model, ddf)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, dec_inputs, enc_outputs, look_ahead_mask, padding_mask):
        att1, att_weights_block1 = self.mha1(dec_inputs, dec_inputs, dec_inputs, look_ahead_mask)
        att1 = self.dropout1(att1)
        out1 = self.layernorm1(att1 + dec_inputs)

        att2, att_weights_block2 = self.mha2(out1, enc_outputs, enc_outputs, padding_mask)
        att2 = self.dropout2(att2)
        out2 = self.layernorm2(att2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, att_weights_block1, att_weights_block2


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ddf, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_embedding = positional_encoding(max_seq_len, d_model)

        self.decode_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ddf, dropout_rate) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, dec_inputs, enc_outputs, look_ahead_mask, padding_mask):
        seq_len = dec_inputs.size(1)
        word_emb = self.embedding(dec_inputs)
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb)

        for i in range(self.num_layers):
            x, att_weights_block1, att_weights_block2 = self.decode_layers[i](x, enc_outputs, look_ahead_mask,
                                                                              padding_mask)

        return x, att_weights_block1, att_weights_block2


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ddf, input_vocab_size, target_vocab_size, max_seq_len,
                 drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, ddf, input_vocab_size, max_seq_len, drop_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, ddf, target_vocab_size, max_seq_len, drop_rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, enc_inputs, dec_inputs, enc_mask, look_ahead_mask, dec_padding_mask):
        enc_outputs = self.encoder(enc_inputs, enc_mask)
        dec_outputs, _, _ = self.decoder(dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask)
        final_outputs = self.final_layer(dec_outputs)
        return final_outputs


transformer = Transformer(num_layers=2, d_model=512, num_heads=8, ddf=1024, input_vocab_size=5000,
                          target_vocab_size=5000, max_seq_len=200)
if training:
    transformer.train()
encoder_input = torch.randint(5000, (32, 100))
decoder_input = torch.randint(5000, (32, 100))
transformer_output = transformer(encoder_input, decoder_input, enc_mask=None, look_ahead_mask=None,
                                 dec_padding_mask=None)
print(transformer_output.shape)
