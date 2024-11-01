# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
计算图像信息熵
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
import numpy as np
import rasterio
from scipy.stats import entropy

def calculate_entropy(data):
    """计算图像的熵."""
    # Flatten the data to a 1D array
    values, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=2)

def process_folder(folder_path):
    """读取文件夹中的所有 .tif 文件并计算每张影像的熵."""
    results = {}
    all_entropy_values = []  # 用于存储所有影像的二维熵
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(folder_path, filename)
            
            # 打开影像文件
            with rasterio.open(file_path) as src:
                # 获取影像的波段数
                band_count = src.count
                entropy_values = []
                
                # 对每个波段计算熵
                for i in range(1, band_count + 1):
                    band_data = src.read(i)
                    band_entropy = calculate_entropy(band_data)
                    entropy_values.append(band_entropy)
                    all_entropy_values.append(band_entropy)  # 添加到总的熵值列表
                
                # 将每张影像的熵值存储在字典中
                results[filename] = entropy_values
                # 计算所有图像的二维熵均值
                average_entropy = np.mean(all_entropy_values)                
    
    return results, average_entropy

ratio_list = []
ratios = np.arange(0.5, 1, 0.02)
for ratio in ratios:
    ratio_percentage = f"{1/ratio:.2f}"
    ratio_list.append(str(ratio_percentage))
print(ratio_list)

for ratio in ratio_list:
    # 使用示例
    folder_path = r'N:\项目数据\花江\语义分割\1-clip_img\建筑_1024_downsample_x' + ratio
    entropy_results, average_entropy = process_folder(folder_path)
    # for filename, entropies in entropy_results.items():
        # print(f"Image: {filename}, Entropies: {entropies}")
    # 输出所有图片的二维熵均值
    print(average_entropy)