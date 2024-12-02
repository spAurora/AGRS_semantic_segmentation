import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.MAE_encoder import MaskedAutoencoderViT
from networks.MAE_decoder_Naive import MAESSDecoderNaive



class MAEViTSegmentation(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 patch_size=16, 
                 band_num=4, 
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
            in_chans=band_num,
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
        self.decoder = MAESSDecoderNaive(
            embed_dim=embed_dim,
            patch_size=patch_size,
            in_chans=band_num,
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
        latent = self.encoder.forward_encoder(x) # latent[0] 16 x 257 x 1280

        # 解码器：生成语义分割结果
        segmentation_output = self.decoder(latent)
        return segmentation_output
