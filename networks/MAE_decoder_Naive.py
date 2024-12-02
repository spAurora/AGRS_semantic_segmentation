import torch.nn as nn
import torch.nn.functional as F

class MAESSDecoderNaive(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans, num_classes):
        super(MAESSDecoderNaive, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes

        # 将编码器的输出还原为每个 patch 的空间分布
        self.linear_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        )

        # 卷积层用于特征变换
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(526, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(128, num_classes, kernel_size=1)

    def unpatchify(self, x):
        """
        将 patch 格式还原为图像格式
        x: [batch_size, num_patches, patch_size**2 * in_chans]
        """
        p = self.patch_size
        batch_size, num_patches, _ = x.shape
        pn_h = pn_w = int(num_patches ** 0.5)  # 假设图像是方形的

        x = x.reshape(batch_size, pn_h, pn_w, p, p, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # -> (batch_size, in_chans, pn_h, p, pn_w, p)
        x = x.reshape(batch_size, self.in_chans, pn_h * p, pn_w * p) # -> (batch_size, in_chans, h, w)
        return x

    def forward(self, x):
        # 去掉 cls_token
        x = x[0]
        x = x[:, 1:, :]

        # 线性变换 + unpatchify
        x = self.linear_proj(x)
        x = self.unpatchify(x)

        # 卷积特征变换
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x