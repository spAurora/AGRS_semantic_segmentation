import torch.nn as nn
import torch.nn.functional as F
import torch

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=13):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            # nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x

class MAESSDecoderFPN(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans, num_classes):
        super(MAESSDecoderFPN, self).__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.down_scale_factor = 1
        self.embed_dim = embed_dim
        self.patch_size = patch_size // self.down_scale_factor

        self.conv0 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 16, 16),
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 8, 8),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 4, 4),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            nn.ConvTranspose2d(2048, 2048, 2, 2),
            nn.Dropout(0.5)
        )

        self.decoder = FPNHEAD()

        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=3, padding=1),
        )

    def change_shape(self, x):
        """
        调整编码器形状
        input x: [batch_size, num_patches, embed_dim]
        output x: [batch_size, embed_dim, pn_h, pn_w]
        """
        batch_size, num_patches, _ = x.shape
        pn_h = pn_w = int(num_patches ** 0.5)

        x = x.reshape(batch_size, self.embed_dim, pn_h, pn_w).contiguous()
        return x

    def forward(self, x):
        x = x[0]
        x = x[:, 1:, :]
        x = self.change_shape(x)

        m = {}
        m[0] = self.conv0(x)  # 256, pn_h*16, pn_w*16
        m[1] = self.conv1(x)  # 512, pn_h*8, pn_w*8
        m[2] = self.conv2(x)  # 1024, pn_h*4, pn_w*4
        m[3] = self.conv3(x)  # 2048, pn_h*2, pn_w*2

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)
        return x