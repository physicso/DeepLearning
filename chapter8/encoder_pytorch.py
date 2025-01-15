import torch
import torch.nn as nn
from positional_encoding_pytorch import positional_encoding
from multi_head_attention_pytorch import MultiHeadAttention

training = False


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


encoder = Encoder(n_layers=2, d_model=512, n_heads=8, ddf=1024, input_vocab_size=5000, max_seq_len=200)
if training:
    encoder.train()
encoder_output = encoder(torch.randint(5000, (32, 100)), None)
print(encoder_output.shape)
