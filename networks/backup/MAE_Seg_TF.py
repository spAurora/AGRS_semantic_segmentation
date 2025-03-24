"""
ref: https://blog.csdn.net/Gu_NN/article/details/125350058
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.MAE_decoder_Naive import MAESSDecoderNaive
from networks.MAE_decoder_ExNaive import MAESSDecoderExNaive

from transformers import ViTMAEForPreTraining, ViTMAEConfig

class MAEViTSegmentation(nn.Module):
    def __init__(self,
                 band_num=4,
                 config_path=r'E:/project_global_populus/MAE_test_250324\3-weights/config.json', # 更换不同encoder
                 num_classes=2):
        super(MAEViTSegmentation, self).__init__()
        
        # 编码器：MAE 的 ViT 部分
        # 加载配置文件
        model_config = ViTMAEConfig.from_json_file(config_path)
        # 初始化模型
        pre_train_model_full = ViTMAEForPreTraining(model_config)
        # 编码器部分
        self.encoder = pre_train_model_full.vit

        # 解码器：用于语义分割
        self.decoder = MAESSDecoderExNaive( # 更换不同decoder
            embed_dim=model_config.hidden_size,
            patch_size=model_config.patch_size,
            in_chans=model_config.num_channels,
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
        latent = self.encoder(x).last_hidden_state # latent

        # print(latent.shape)

        # 解码器：生成语义分割结果
        segmentation_output = self.decoder(latent)
        return segmentation_output
