# -*- coding: utf-8 -*-

"""
批量修改图片大小
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
Ghent University
Haoyu.Wang@ugent.be
"""

from PIL import Image
import fnmatch
import os

img_path = r'D:\github_repository\ChaIR\Dehazing\OTS\results\ChaIR\test\LV2\256'
output_path = r'D:\github_repository\ChaIR\Dehazing\OTS\results\ChaIR\test\LV2'

listpic = fnmatch.filter(os.listdir(img_path), '*.tif')

for img in listpic:
    img_full_path = img_path + '/' + img
    output_full_path = output_path + '/' + img

    # 读取原始图片
    original_image = Image.open(img_full_path)

    # 指定目标大小
    target_size = (384, 384)  # 宽度和高度设置

    # 调整图片大小
    resized_image = original_image.resize(target_size)

    # 保存调整大小后的图片
    resized_image.save(output_full_path)
