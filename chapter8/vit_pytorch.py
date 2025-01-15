import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        x = self.proj(inputs)
        x = x.flatten(2).transpose(1, 2)
        return x


class AddClassTokenAddPosEmbed(nn.Module):
    def __init__(self, embed_dim=768, num_patches=196):
        super(AddClassTokenAddPosEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, inputs):
        B, N, _ = inputs.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, inputs], dim=1)
        x = x + self.pos_embed
        return x


emb = PatchEmbed()
emb2 = AddClassTokenAddPosEmbed()
out = emb2(emb(torch.rand(10, 3, 224, 224)))
print(out.shape)


# In a more concise way, the Q, K, and V matrices are represented together.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, use_bias=False,
                 attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.scale = self.depth ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=use_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, inputs):
        B, N, C = inputs.shape

        qkv = self.qkv(inputs).reshape(B, N, 3, self.num_heads, self.depth).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_model, diff, drop=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, diff)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(diff, d_model)

    def forward(self, inputs):
        x = F.gelu(self.fc1(inputs))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads=8, use_bias=False,
                 mlp_drop_ratio=0.1, attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1, drop_path_ratio=0.1):
        super(Encoder, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads=num_heads, use_bias=use_bias, attn_drop_ratio=attn_drop_ratio,
                                       proj_drop_ratio=proj_drop_ratio)
        self.drop_path = nn.Dropout(drop_path_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4, drop=mlp_drop_ratio)

    def forward(self, inputs):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


encoder = Encoder(d_model=768)
out = encoder(torch.rand(10, 197, 768))
print(out.shape)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 n_layers=2, num_heads=8, use_bias=True,
                 mlp_drop_ratio=0.1, attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1, drop_path_ratio=0.1,
                 num_classes=1000):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_pos_embed = AddClassTokenAddPosEmbed(embed_dim=embed_dim, num_patches=num_patches)

        self.pos_drop = nn.Dropout(proj_drop_ratio)

        self.encoders = nn.ModuleList([
            Encoder(d_model=embed_dim, num_heads=num_heads, use_bias=use_bias, mlp_drop_ratio=mlp_drop_ratio,
                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio, drop_path_ratio=drop_path_ratio)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs):
        x = self.patch_embed(inputs)
        x = self.cls_pos_embed(x)
        x = self.pos_drop(x)

        for encoder in self.encoders:
            x = encoder(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


vit = VisionTransformer()
out = vit(torch.rand(10, 3, 224, 224))
print(out.shape)
