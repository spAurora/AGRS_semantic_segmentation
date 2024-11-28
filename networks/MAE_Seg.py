import torch
import torch.nn as nn
import torch.nn.functional as F
from MAE import MaskedAutoencoderViT  # 假设 MAE-ViT 的实现文件名为 mae.py

class SemanticSegmentationDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans, num_classes):
        super(SemanticSegmentationDecoder, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes

        # 将编码器的输出还原为每个 patch 的空间分布
        self.linear_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

        # 卷积层用于逐步上采样到原始分辨率
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)

    def unpatchify(self, x):
        """
        将 patch 格式还原为图像格式
        x: [batch_size, num_patches, patch_size**2 * in_chans]
        """
        p = self.patch_size
        batch_size, num_patches, _ = x.shape
        h = w = int(num_patches ** 0.5)  # 假设图像是方形的

        x = x.reshape(batch_size, h, w, p, p, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, self.in_chans, h * p, w * p)
        return x

    def forward(self, x):
        # 去掉 cls_token
        x = x[:, 1:, :]

        # 线性变换 + unpatchify
        x = self.linear_proj(x)
        x = self.unpatchify(x)

        # 卷积解码
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # 输出类别概率

        return nn.Sigmoid(x)

class MAEViTSegmentation(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 in_chans=4, 
                 embed_dim=1280, 
                 depth=32, 
                 num_heads=16, 
                 decoder_embed_dim=512, 
                 num_classes=2):
        super(MAEViTSegmentation, self).__init__()
        
        # 编码器：MAE 的 ViT 部分
        self.encoder = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=0,  # 解码器部分不需要 MAE 自带的解码器
            decoder_num_heads=0,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm
        )

        # 解码器：用于语义分割
        self.decoder = SemanticSegmentationDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes
        )

    def freeze_encoder(self):
        # 冻结编码器所有层的参数
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        # 解冻编码器所有层的参数
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 编码器：提取特征
        latent = self.encoder.forward_encoder(x)

        # 解码器：生成语义分割结果
        segmentation_output = self.decoder(latent)
        return segmentation_output
