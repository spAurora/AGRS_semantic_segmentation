# -*- coding: utf-8 -*-
"""
code by Jiudan ZHANG
Chinese Academy of Sciences
Jiudan.Zhang@ugent.be
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.elu, inplace=True)

class Dblock(nn.Module): # multi-dilated eonvolution module
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(
            int(channel/2), channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=3, padding=3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        out = dilate1_out + dilate2_out + dilate3_out
        return out


class DUNet_Plus(nn.Module):
    def __init__(self, num_classes=3, band_num=3):
        super(DUNet_Plus, self).__init__()

        vgg13 = models.vgg13(pretrained=False)

        self.conv1 = nn.Conv2d(band_num, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = vgg13.features[2]
        self.conv3 = vgg13.features[5]
        self.conv4 = vgg13.features[7]
        self.conv5 = vgg13.features[10]
        self.conv6 = vgg13.features[12]

        self.dilate_center = Dblock(512)

        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)

        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, num_classes, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.ELU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.ELU(inplace=True)
        )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        stage1 = nonlinearity(self.conv2(nonlinearity(self.conv1(x))))
        stage2 = nonlinearity(self.conv4(
            nonlinearity(self.conv3(self.max_pool(stage1)))))
        stage3 = nonlinearity(self.conv6(
            nonlinearity(self.conv5(self.max_pool(stage2)))))

        out = self.dilate_center(self.max_pool(stage3))

        out = self.up3(torch.cat((self.trans3(out), stage3), 1))
        out = self.up2(torch.cat((self.trans2(out), stage2), 1))
        out = self.up1(torch.cat((self.trans1(out), stage1), 1))

        out = self.conv_last(out)

        return out
