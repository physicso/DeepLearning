import torch
import torch.nn as nn
import torch.nn.functional as F


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


mha = MultiHeadAttention(d_model=512, num_heads=8)
y = torch.randn(1, 50, 512)
output, att = mha(y, y, y, mask=None)
print(output.shape, att.shape)
