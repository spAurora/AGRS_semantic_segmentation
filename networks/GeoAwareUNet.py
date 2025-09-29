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

            #lon_lat[i, 0]=0 if lon_lat[i, 0] < 90 else 1
            #lon_lat[i, 1]=0 if lon_lat[i, 1] < 41 else 1

            # print(lon_lat[i, 0], lon_lat[i, 1])
        
        return lon_lat

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 修正通道数计算
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        # 修正输入通道数为拼接后的正确值
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不一致的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 修正通道数设置
        self.up1 = Up(512, 256)  # 输入512，输出256
        self.up2 = Up(256, 128)  # 输入256，输出128
        self.up3 = Up(128, 64)   # 输入128，输出64
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x, enc4, enc3, enc2):
        x = self.up1(x, enc4)  # x: [B, 256, H, W]
        x = self.up2(x, enc3)  # x: [B, 128, H, W]
        x = self.up3(x, enc2)  # x: [B, 64, H, W]
        return self.final_conv(x)

class GeoAwareUNet(nn.Module):
    def __init__(self, band_num=9, num_classes=1, geo_feature_dim=2):
        super().__init__()
        
        # UNet主干部分
        self.encoder = Encoder(band_num-1)  # 使用前8个波段
        self.bottleneck = Bottleneck(512, 512)
        self.decoder = Decoder(512, num_classes)

        self.geo_decoder = GeoDecoder()
        
        # 地理特征处理分支
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1024),
            nn.ReLU()
        )
        
        # 自适应FiLM参数生成
        self.film_generator = nn.Linear(512, 1024)
        
    def forward(self, x):
        # 分离图像数据和地理信息
        image = x[:, :-1, :, :]  # 前8个波段
        geo_channel = x[:, -1:, :, :]  # 第9个波段是地理信息

        # 提取图像特征
        enc1, enc2, enc3, enc4 = self.encoder(image)
        bottleneck = self.bottleneck(enc4)
        
        # 处理地理特征
        lon_lat = self.geo_decoder(geo_channel)
        geo_context = self.geo_mlp(lon_lat)
        
        # 结合图像和地理特征生成FiLM参数
        combined = self.film_generator(bottleneck.mean(dim=[2,3])) + geo_context
        gamma, beta = torch.chunk(combined, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # 特征调制
        modulated_bottleneck = gamma * bottleneck + beta
        
        # 解码
        output = self.decoder(modulated_bottleneck, enc3, enc2, enc1)
        return output