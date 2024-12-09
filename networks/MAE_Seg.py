import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.MAE_encoder import mae_vit_base_patch16_dec512d8b_populus, mae_vit_huge_patch16_populus, mae_vit_base_patch16_dec512d8b_populus_small
from networks.MAE_decoder_Naive import MAESSDecoderNaive
from networks.MAE_decoder_FPN import MAESSDecoderFPN

def get_pretrained_mae_model(model_type, **kwargs):
    if model_type == "mae_vit_base_patch16_dec512d8b_populus":
        return mae_vit_base_patch16_dec512d8b_populus(**kwargs)
    elif model_type == "mae_vit_huge_patch16_populus":
        return mae_vit_huge_patch16_populus(**kwargs)
    elif model_type == "mae_vit_base_patch16_dec512d8b_populus_small":
        return mae_vit_base_patch16_dec512d8b_populus_small(**kwargs)
    else:
        raise ValueError("Unsupported model type")

class MAEViTSegmentation(nn.Module):
    def __init__(self,
                 band_num=4,
                 model_type="mae_vit_base_patch16_dec512d8b_populus_small", # 更换不同encoder
                 num_classes=2):
        super(MAEViTSegmentation, self).__init__()
        
        # 编码器：MAE 的 ViT 部分
        self.encoder = get_pretrained_mae_model(model_type)

        # 解码器：用于语义分割
        self.decoder = MAESSDecoderNaive(
            embed_dim=self.encoder.embed_dim,
            patch_size=self.encoder.patch_size,
            in_chans=self.encoder.in_chans,
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
        latent = self.encoder.forward_encoder(x) # latent[0] batch_size x patch_num+1 x embed_dim

        # 解码器：生成语义分割结果
        segmentation_output = self.decoder(latent)
        return segmentation_output
