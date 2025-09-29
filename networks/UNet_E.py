import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):                       ## 基础特征提取单元 
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):                     ## 下采样模块（编码器），这是UNet的左侧部分，负责逐步提取和压缩特征​：
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):                    ##  上采样模块（解码器），这是UNet的右侧部分，负责逐步恢复特征图尺寸并进行精确定位​：
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)



'''高程引导的注意力模块'''      # 这是本模型最创新的部分，它让模型能够关注高程信息重要的区域

class ElevationGuidedAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 高程空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 高程值到注意力权重的映射
        self.elevation_weight_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, elevation_map):

        # 通道注意力，关注哪些特征通道对当前任务更重要
        ca = self.channel_attention(x)
        
        # 空间注意力（基于高程），基于高程图，关注哪些空间位置更重要
        sa = self.spatial_attention(elevation_map)
        
        # 高程值权重（全局调节），根据平均高程值生成全局权重
        avg_elevation = elevation_map.mean(dim=[2, 3], keepdim=True)
        elevation_weight = self.elevation_weight_mapper(avg_elevation.permute(0, 2, 3, 1))
        elevation_weight = elevation_weight.permute(0, 3, 1, 2)
        
        # 组合注意力
        attention = ca * sa * elevation_weight
        # print(ca.shape, sa.shape, elevation_map.shape, x.shape, attention.shape)
        return x * attention


'''高程特征提取器'''       #专门用于提取高程数据的多尺度特征：
class EnhancedElevationExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 多尺度高程特征提取
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2, dilation=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, dilation=2),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, dilation=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, dilation=3),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=3, dilation=3),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=3, dilation=3),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.attention = ElevationGuidedAttention(256)

    def forward(self, elevation):
        # 多尺度特征提取
        features = []
        for conv in self.conv_layers:
            features.append(conv(elevation))
        
        # print('features', features[0].shape, features[1].shape,features[2].shape)

        fused = torch.cat(features, dim=1)
        x = self.fusion(fused)
        
        # 应用高程引导的注意力
        elevation_down_x8 = F.interpolate(elevation, scale_factor=0.125, mode='bilinear', align_corners=False)
        # print('EnhancedElevationExtractor',x.shape, elevation_down_x8.shape)
        x = self.attention(x, elevation_down_x8)
        return x

'''改进的UNet模型'''
class UNetWithElevationAttention(nn.Module):
    def __init__(self, band_num, num_classes, bilinear=True, ifVis=False):
        super().__init__()
        self.ifVis = ifVis
        self.n_channels = band_num
        self.n_classes = num_classes
        self.bilinear = bilinear

        # 主路径（处理前4个波段）
        self.inc = DoubleConv(4, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # 高程注意力路径
        self.elevation_extractor = EnhancedElevationExtractor()
        
        # 跨层注意力注入点
        self.attention_injection1 = ElevationGuidedAttention(64)
        self.attention_injection2 = ElevationGuidedAttention(128)
        self.attention_injection3 = ElevationGuidedAttention(256)
        
        # 融合后的下采样
        factor = 2 if bilinear else 1
        self.down4 = Down(512 + 256, 1024 // factor)
        
        # 上采样路径
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出前的高程注意力
        self.final_attention = ElevationGuidedAttention(64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # 分离输入数据
        spectral = x[:, :4, :, :]
        elevation = x[:, 4:, :, :]
        
        # 主路径处理（逐层注入高程注意力）
        x1 = self.inc(spectral)
        elevation1 = elevation
        x1 = self.attention_injection1(x1, elevation1)  # 注入高程注意力
        
        x2 = self.down1(x1)
        elevation2 = F.interpolate(elevation1, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2 = self.attention_injection2(x2, elevation2)  # 注入高程注意力
        
        x3 = self.down2(x2)
        elevation3 = F.interpolate(elevation2, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3 = self.attention_injection3(x3, elevation3)  # 注入高程注意力
        
        x4 = self.down3(x3)
        
        # 高程特征提取
        elev_feat = self.elevation_extractor(elevation)
        # print('elev_feat', x4.shape, elev_feat.shape)
        
        # 特征融合
        x5_input = torch.cat([x4, elev_feat], dim=1)
        x5 = self.down4(x5_input)
        
        # 上采样路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 最终的高程注意力调节
        x = self.final_attention(x, elevation)
        
        logits = self.outc(x)
        if self.ifVis:
            return logits, (x3, elev_feat, elevation)
        else:
            return logits