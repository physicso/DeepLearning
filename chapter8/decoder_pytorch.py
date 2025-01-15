import torch
import torch.nn as nn
from positional_encoding_pytorch import positional_encoding
from multi_head_attention_pytorch import MultiHeadAttention
from encoder_pytorch import feed_forward_network, Encoder

training = False


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


encoder = Encoder(n_layers=2, d_model=512, n_heads=8, ddf=1024, input_vocab_size=5000, max_seq_len=200)
encoder_output = encoder(torch.randint(5000, (32, 100)), None)
decoder = Decoder(num_layers=2, d_model=512, num_heads=8, ddf=1024, target_vocab_size=5000, max_seq_len=200)
if training:
   decoder.train()
decoder_output = decoder(torch.randint(5000, (32, 100)), encoder_output, None, None)
print(decoder_output[0].shape)
