import tensorflow as tf
import keras
from keras import layers


class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = tf.keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size,
                                           padding='SAME')

    def call(self, inputs):
        B, H, W, C = inputs.shape
        x = self.proj(inputs)
        x = tf.reshape(x, [B, self.num_patches, self.embed_dim])
        return x


class AddClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=768, num_patches=196):
        super(AddClassTokenAddPosEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.cls_token = self.add_weight(name="cls", shape=[1, 1, self.embed_dim],
                                         initializer=keras.initializers.Zeros(), trainable=True, dtype=tf.float32)

        self.pos_embed = self.add_weight(name="pos_embed", shape=[1, self.num_patches + 1, self.embed_dim],
                                         initializer=keras.initializers.RandomNormal(), trainable=True,
                                         dtype=tf.float32)

    def call(self, inputs):
        B, _, _ = inputs.shape
        cls_token = tf.broadcast_to(self.cls_token, shape=[B, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)
        x = x + self.pos_embed
        return x


emb = PatchEmbed()
emb2 = AddClassTokenAddPosEmbed()
out = emb2(emb(tf.random.uniform((10, 224, 224, 3))))
print(out.shape)


# In a more concise way, the Q, K, and V matrices are represented together.
class MutilHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads=8, use_bias=False, attn_drop_ratio=0.1, proj_drop_ratio=0.1):
        super(MutilHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.scale = self.depth ** -0.5

        self.qkv = layers.Dense(d_model * 3, use_bias=use_bias)
        self.attn_drop = layers.Dropout(attn_drop_ratio)

        self.proj = layers.Dense(d_model)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

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


class Encoder(layers.Layer):
    def __init__(self, d_model, num_heads=8, use_bias=False, mlp_drop_ratio=0.1, attn_drop_ratio=0.1,
                 proj_drop_ratio=0.1, drop_path_ratio=0.1):
        super(Encoder, self).__init__()

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = MutilHeadAttention(d_model, num_heads=num_heads, use_bias=use_bias, attn_drop_ratio=attn_drop_ratio,
                                       proj_drop_ratio=proj_drop_ratio)

        self.drop_path = layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1))

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(d_model, d_model * 4, drop=mlp_drop_ratio)

    def call(self, inputs, training):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


encoder = Encoder(d_model=768)
out = encoder(tf.random.uniform((10, 197, 768)))
print(out.shape)


class VisionTransformer(keras.Model):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, n_layers=2, num_heads=8, use_bias=True,
                 mlp_drop_ratio=0.1, attn_drop_ratio=0.1, proj_drop_ratio=0.1, drop_path_ratio=0.1, num_classes=1000):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.use_bias = use_bias

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_pos_embed = AddClassTokenAddPosEmbed(embed_dim=embed_dim,
                                                      num_patches=num_patches)

        self.pos_drop = layers.Dropout(drop_path_ratio)

        self.encoders = [
            Encoder(d_model=embed_dim, num_heads=num_heads, use_bias=use_bias, mlp_drop_ratio=mlp_drop_ratio,
                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=proj_drop_ratio, drop_path_ratio=drop_path_ratio)
            for _ in range(self.n_layers)]

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.out = layers.Dense(num_classes)

    def call(self, inputs, training):
        x = self.patch_embed(inputs)
        x = self.cls_pos_embed(x)
        x = self.pos_drop(x, training=training)

        for encoder in self.encoders:
            x = encoder(x, training=training)

        x = self.norm(x)
        x = self.out(x[:, 0])
        return x


vit = VisionTransformer()
out = vit(tf.random.uniform((10, 224, 224, 3)))
print(out.shape)
