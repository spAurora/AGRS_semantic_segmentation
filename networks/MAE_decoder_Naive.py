import torch.nn as nn
import torch.nn.functional as F

class MAESSDecoderNaive(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans, num_classes):
        super(MAESSDecoderNaive, self).__init__()
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
        x = x[0]
        x = x[:, 1:, :]

        # 线性变换 + unpatchify
        x = self.linear_proj(x)
        x = self.unpatchify(x)

        # 卷积解码
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # 输出类别概率

        return x