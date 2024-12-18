# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
tif影像无损压缩
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import os
import rasterio
from rasterio.enums import Compression

# 输入文件夹和输出文件夹路径
input_folder = r'N:\项目数据\花江\语义分割\1-clip_img\建筑_1024_downsample_x8'
output_folder = r'N:\项目数据\花江\语义分割\1-clip_img\建筑_1024_downsample_x8\LZW'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, filename)

        # 读取TIFF影像
        with rasterio.open(input_filepath) as src:
            # 读取影像数据
            data = src.read()
            # 获取影像元数据
            meta = src.meta.copy()
            # 修改压缩方式为无损压缩
            meta.update(compress='LZW')  # 使用 LZW 压缩

            # 写入压缩后的影像
            with rasterio.open(output_filepath, 'w', **meta) as dst:
                dst.write(data)

print("无损压缩完成，所有影像已保存到指定文件夹。")
