# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
计算图像NDVI
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
import rasterio
import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

def calculate_ndvi(input_folder, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹下的所有tif文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_filepath = os.path.join(input_folder, filename)
            
            # 读取哨兵二号影像数据
            with rasterio.open(input_filepath) as src:
                # 检查波段数是否为4（R, G, B, NIR）
                if src.count < 4:
                    print(f"文件 {filename} 不包含足够的波段数 (至少需要4个波段).")
                    continue
                
                # 读取R波段（通常是第1个波段）和NIR波段（通常是第4个波段）
                red_band = src.read(1).astype('float32')
                nir_band = src.read(4).astype('float32')
                
                # 计算NDVI，防止除零错误
                ndvi = np.where((nir_band + red_band) == 0, 0, (nir_band - red_band) / (nir_band + red_band))
                
                # 创建输出文件路径
                output_filepath = os.path.join(output_folder, f"NDVI_{filename}")
                
                # 将NDVI写入到GeoTIFF文件中
                profile = src.profile
                profile.update(dtype=rasterio.float32, count=1, compress='lzw')
                
                with rasterio.open(output_filepath, 'w', **profile) as dst:
                    dst.write(ndvi, 1)
            
            print(f"成功处理并保存 NDVI 图像: {output_filepath}")

# 示例用法
input_folder = r"E:\paper_lishuo_new\new-fig-5\0-clip_img\1-clip_img"  # 替换为实际的输入文件夹路径
output_folder = r"E:\paper_lishuo_new\new-fig-5\0-clip_img\4-clip_img_ndvi"  # 替换为实际的输出文件夹路径
calculate_ndvi(input_folder, output_folder)