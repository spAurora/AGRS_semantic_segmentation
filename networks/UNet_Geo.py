import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoDecoder(nn.Module):
    """地理信息解码器"""
    def __init__(self):
        super(GeoDecoder, self).__init__()
        
    def forward(self, geo_channel):
        """
        从最后一个波段解码经纬度信息
        参数:
            geo_channel: [batch_size, 1, height, width]
        返回:
            lon_lat: [batch_size, 2]
        """
        batch_size = geo_channel.size(0)
        lon_lat = torch.zeros((batch_size, 2), device=geo_channel.device)
        
        for i in range(batch_size):
            # 解码经度（前15像素）
            lon_digits = ''.join(str(int(x)) for x in geo_channel[i, 0, 0, :15])
            lon_int = int(lon_digits[:2])
            lon_frac = int(lon_digits[2:])
            
            # 解码纬度（后15像素）
            lat_digits = ''.join(str(int(x)) for x in geo_channel[i, 0, 0, 15:30])
            lat_int = int(lat_digits[:2])
            lat_frac = int(lat_digits[2:])
            
            # 符号位处理
            sign_byte = int(geo_channel[i, 0, 0, 30])
            lon_sign = -1 if (sign_byte & 0b00000001) else 1
            lat_sign = -1 if (sign_byte & 0b00000010) else 1
            
            # 组合经度
            lon_str = f"{lon_int}.{lon_frac}"
            lon = lon_sign * float(lon_str)
            
            # 组合纬度
            lat_str = f"{lat_int}.{lat_frac}"
            lat = lat_sign * float(lat_str)

            lon_lat[i, 0] = lon_sign * (lon)
            lon_lat[i, 1] = lat_sign * (lat)

            # print(lon_lat[i, 0], lon_lat[i, 1])
        
        return lon_lat

class GeoEncoder(nn.Module):
    """地理信息编码器"""
    def __init__(self, hidden_dim=32):
        super(GeoEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1024),
            nn.ReLU()
        )
        
    def forward(self, lon_lat):
        return self.mlp(lon_lat)

class FiLMLayer(nn.Module):
    """FiLM条件化层"""
    def __init__(self, feature_dim, condition_dim):
        super(FiLMLayer, self).__init__()
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class DoubleConv(nn.Module):
    """双卷积块"""
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
    """下采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块"""
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
    """输出层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GeoUNet(nn.Module):
    """地理信息融合UNet"""
    def __init__(self, band_num, num_classes, bilinear=True, ifVis=False):
        super(GeoUNet, self).__init__()
        self.ifVis = ifVis
        self.n_channels = band_num - 1  # 最后一个通道是地理信息
        self.n_classes = num_classes
        self.bilinear = bilinear

        # 地理信息处理模块
        self.geo_decoder = GeoDecoder()
        self.geo_encoder = GeoEncoder(hidden_dim=32)
        
        # UNet主干网络
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 地理信息融合层
        self.film = FiLMLayer(1024 // factor, 1024)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # 分离图像数据和地理信息
        img_data = x[:, :-1, :, :]  # 前n_channels-1个通道
        geo_channel = x[:, -1:, :, :]  # 最后一个通道是地理信息
        
        # 解码和编码地理信息
        lon_lat = self.geo_decoder(geo_channel)
        geo_feature = self.geo_encoder(lon_lat)
        
        # UNet编码路径
        x1 = self.inc(img_data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 在瓶颈层融合地理信息
        x5 = self.film(x5, geo_feature)
        
        # UNet解码路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if self.ifVis:
            return logits, x3  # 协同输出可视化信息
        else:
            return logits