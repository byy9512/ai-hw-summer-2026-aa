import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits an image into patches and linearly projects each into an embedding."""

    def __init__(self, image_size: int = 28, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoderModel(nn.Module):
    """ViT-style transformer encoder for MNIST digit classification."""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])  # take the class token
        return self.head(x)
