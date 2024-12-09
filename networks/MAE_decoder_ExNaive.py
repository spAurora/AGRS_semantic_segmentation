import torch.nn as nn
import torch.nn.functional as F

class MAESSDecoderExNaive(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans, num_classes):
        super(MAESSDecoderExNaive, self).__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.head = nn.ConvTranspose2d(embed_dim, num_classes, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        # 去掉 cls_token
        batch_size, _ = x.shape
        x = x[0]
        x = x[:, 1:, :] # B, PN, embed_dim

        x = x.transpose(1, 2).reshape(batch_size, -1, 256, 256)
        x = self.head(x)
        
        return x