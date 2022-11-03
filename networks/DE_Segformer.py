# -*- coding: utf-8 -*-
"""
DE-Segformer
解码器增强的Segformer
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

from math import sqrt
from functools import partial
import torch
from torch import nn, einsum


from einops import rearrange

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 2, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class DE_Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        band_num = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.decoder_dim = decoder_dim

        self.mit = MiT(
            channels = band_num,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.decode_stage_1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU()
        ) for i, dim in enumerate(dims)])

        self.downsampleConv = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, 2, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU())
        
        self.upsampleTransConv1 = self.UpSampleTransConv(1)
        self.upsampleTransConv2 = self.UpSampleTransConv(2)
        self.upsampleTransConv3 = self.UpSampleTransConv(3)

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(decoder_dim*4, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 4, 2, 1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(), 
            #nn.Upsample(scale_factor=2), #无TC实验
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def UpSampleTransConv(self, scale):
        return nn.Sequential(
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, (2 ** scale)*2, 2 ** scale, 2 ** (scale-1), bias=False),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU())      

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [decode_stage_1(output) for output, decode_stage_1 in zip(layer_outputs, self.decode_stage_1)] # 通道规整
        
        fused_merge = []  # 多尺度特征图融合 Merge Block
        fused_merge.append(fused[0])
        for i in range(0,3):
            fused_downConv = self.downsampleConv(fused_merge[i])
            fused_merge.append(fused_downConv + fused[i+1])
            #fused_merge.append(fused[i+1]) # 无MB实验
 
        fused_merge[1] = self.upsampleTransConv1(fused_merge[1]) #上采样到H/2*W/2
        fused_merge[2] = self.upsampleTransConv2(fused_merge[2])
        fused_merge[3] = self.upsampleTransConv3(fused_merge[3])
        
        fused_merge = torch.cat(fused_merge, dim = 1)  # 拼接
        fused_merge = self.to_segmentation(fused_merge) # 上采样到H*W   

        return torch.sigmoid(fused_merge)

        
