# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
计算一组图像的NDVI，并输出它们的极大值分布图
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import os
import numpy as np
import rasterio

def calculate_max_ndvi(input_folder, output_filepath):
    # 初始化最大NDVI数组为None
    max_ndvi = None

    # 读取所有影像，计算并记录各影像对应像素位置的NDVI值
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_filepath = os.path.join(input_folder, filename)
            
            with rasterio.open(input_filepath) as src:
                # 检查波段数是否为4（R, G, B, NIR）
                if src.count < 4:
                    print(f"文件 {filename} 不包含足够的波段数 (至少需要4个波段).")
                    continue
                
                # 读取R波段（通常是第1个波段）和NIR波段（通常是第4个波段）
                red_band = src.read(1).astype('float32')
                nir_band = src.read(4).astype('float32')
                
                # 计算NDVI
                ndvi = np.where((nir_band + red_band) == 0, 0, (nir_band - red_band) / (nir_band + red_band))
                
                # 如果这是第一个影像，初始化max_ndvi数组
                if max_ndvi is None:
                    max_ndvi = ndvi
                else:
                    # 更新max_ndvi数组，取每个位置的NDVI最大值
                    max_ndvi = np.maximum(max_ndvi, ndvi)

    # 如果没有有效的NDVI数据，退出程序
    if max_ndvi is None:
        print("没有有效的NDVI数据生成最大NDVI图像.")
        return
    
    # 使用第一个影像的元数据作为输出NDVI图像的元数据
    with rasterio.open(input_filepath) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        
        # 将最大NDVI值写入到新的GeoTIFF文件中
        with rasterio.open(output_filepath, 'w', **profile) as dst:
            dst.write(max_ndvi, 1)
    
    print(f"成功生成并保存最大NDVI图像: {output_filepath}")

# 示例用法
input_folder = r"E:\paper_lishuo_new\new-fig-5\0-clip_img\1-clip_img"  # 替换为实际的输入文件夹路径
output_filepath = r"E:\paper_lishuo_new\new-fig-5\0-clip_img\5-clip_img_max_ndvi\1_MAX_NDVI.tif"  # 替换为实际的输出文件路径
calculate_max_ndvi(input_folder, output_filepath)