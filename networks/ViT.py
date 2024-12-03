import torch
import torch.nn as nn
from torchvision import transforms

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = self.flatten(x).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4 * embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=21, band_num=3, img_size=256, patch_size=8, embed_dim=768, num_heads=12, num_layers=12):
        super(ViTSegmentation, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, band_num, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.head = nn.ConvTranspose2d(embed_dim, num_classes, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        patches = self.patch_embed(x)  # (B, num_patches, embed_dim)
        encoded_patches = self.transformer_encoder(patches)  # (B, num_patches, embed_dim)
        h, w = height // self.patch_embed.patch_size, width // self.patch_embed.patch_size
        encoded_patches = encoded_patches.transpose(1, 2).reshape(batch_size, -1, h, w)  # (B, embed_dim, H/P, W/P)
        out = self.head(encoded_patches)  # (B, num_classes, H, W)
        return out