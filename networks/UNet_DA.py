import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

'''双模态注意力模块'''
class DualModalityAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 热红外空间注意力
        self.thermal_spatial = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 高程空间注意力
        self.elevation_spatial = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 热红外权重映射
        self.thermal_weight_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 高程权重映射
        self.elevation_weight_mapper = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, thermal_map, elevation_map):
        # 通道注意力
        ca = self.channel_attention(x)
        
        # 热红外空间注意力
        thermal_sa = self.thermal_spatial(thermal_map)
        
        # 高程空间注意力
        elevation_sa = self.elevation_spatial(elevation_map)
        
        # 热红外全局权重
        avg_thermal = thermal_map.mean(dim=[2,3], keepdim=True)
        thermal_weight = self.thermal_weight_mapper(avg_thermal.permute(0,2,3,1))
        thermal_weight = thermal_weight.permute(0,3,1,2)
        
        # 高程全局权重
        avg_elevation = elevation_map.mean(dim=[2,3], keepdim=True)
        elevation_weight = self.elevation_weight_mapper(avg_elevation.permute(0,2,3,1))
        elevation_weight = elevation_weight.permute(0,3,1,2)
        
        # 组合注意力
        thermal_attention = thermal_sa * thermal_weight
        elevation_attention = elevation_sa * elevation_weight
        
        combined_attention = ca * (self.alpha * thermal_attention + (1-self.alpha) * elevation_attention)
        
        return x * combined_attention

'''双模态特征提取器'''
class DualModalityExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 热红外特征提取
        self.thermal_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 高程特征提取
        self.elevation_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.attention = DualModalityAttention(256)

    def forward(self, thermal, elevation):
        # 提取热红外特征
        thermal_feat = self.thermal_extractor(thermal)
        
        # 提取高程特征
        elevation_feat = self.elevation_extractor(elevation)
        
        # 特征融合
        fused = torch.cat([thermal_feat, elevation_feat], dim=1)
        x = self.fusion(fused)
        
        # 应用双模态注意力
        thermal_down = F.interpolate(thermal, scale_factor=0.125, mode='bilinear', align_corners=False)
        elevation_down = F.interpolate(elevation, scale_factor=0.125, mode='bilinear', align_corners=False)
        x = self.attention(x, thermal_down, elevation_down)
        
        return x

'''改进的UNet模型'''
class UNetWithDualAttention(nn.Module):
    def __init__(self, band_num, num_classes, bilinear=True):
        super().__init__()
        self.n_channels = band_num
        self.n_classes = num_classes
        self.bilinear = bilinear

        # 主路径（处理前N-2个波段）
        self.inc = DoubleConv(band_num-2, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # 双模态特征提取
        self.dual_modality_extractor = DualModalityExtractor()
        
        # 跨层注意力注入点
        self.attention_injection1 = DualModalityAttention(64)
        self.attention_injection2 = DualModalityAttention(128)
        self.attention_injection3 = DualModalityAttention(256)
        
        # 融合后的下采样
        factor = 2 if bilinear else 1
        self.down4 = Down(512 + 256, 1024 // factor)
        
        # 上采样路径
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出前的双模态注意力
        self.final_attention = DualModalityAttention(64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # 分离输入数据
        spectral = x[:, :-2, :, :]  # 前N-2个波段
        thermal = x[:, -2:-1, :, :]  # 倒数第二个波段（热红外）
        elevation = x[:, -1:, :, :]  # 最后一个波段（高程）
        
        # 主路径处理（逐层注入双模态注意力）
        x1 = self.inc(spectral)
        x1 = self.attention_injection1(x1, thermal, elevation)
        
        x2 = self.down1(x1)
        thermal_down1 = F.interpolate(thermal, scale_factor=0.5, mode='bilinear', align_corners=False)
        elevation_down1 = F.interpolate(elevation, scale_factor=0.5, mode='bilinear', align_corners=False)
        x2 = self.attention_injection2(x2, thermal_down1, elevation_down1)
        
        x3 = self.down2(x2)
        thermal_down2 = F.interpolate(thermal_down1, scale_factor=0.5, mode='bilinear', align_corners=False)
        elevation_down2 = F.interpolate(elevation_down1, scale_factor=0.5, mode='bilinear', align_corners=False)
        x3 = self.attention_injection3(x3, thermal_down2, elevation_down2)
        
        x4 = self.down3(x3)
        
        # 双模态特征提取
        modality_feat = self.dual_modality_extractor(thermal, elevation)
        
        # 特征融合
        x5_input = torch.cat([x4, modality_feat], dim=1)
        x5 = self.down4(x5_input)
        
        # 上采样路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 最终的双模态注意力调节
        x = self.final_attention(x, thermal, elevation)
        
        logits = self.outc(x)
        return logits