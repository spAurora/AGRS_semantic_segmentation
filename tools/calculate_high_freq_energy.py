# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
计算图像归一化高频信息量
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2

def analyze_band_high_frequency_energy(band):
    # 进行快速傅里叶变换 (FFT)
    f_transform = np.fft.fft2(band)
    f_transform_shifted = np.fft.fftshift(f_transform)  # 将低频移动到中心

    # 计算频谱幅值
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # 设置一个半径，分离高频成分
    rows, cols = band.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(rows, cols) // 8  # 半径设为图像尺寸的1/4，表示高频区域
    high_pass_mask = np.ones((rows, cols), np.uint8)
    cv2.circle(high_pass_mask, (ccol, crow), radius, 0, -1)  # 掩膜：保留高频

    # 应用掩膜来获取高频成分
    high_freq_component = f_transform_shifted * high_pass_mask
    high_freq_magnitude = np.abs(high_freq_component)

    # 计算高频能量
    high_freq_energy = np.sum(high_freq_magnitude**2)

    # 归一化高频能量（除以总像素数）
    normalized_high_freq_energy = high_freq_energy / (rows * cols)
    
    return normalized_high_freq_energy, magnitude_spectrum, high_freq_magnitude

def process_multiband_images_in_folder(folder_path):
    high_freq_energies = {}

    # 遍历文件夹中的所有.tif文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取多波段图像
            with rasterio.open(file_path) as src:
                band_energies = []
                for i in range(1, src.count + 1):  # 遍历每个波段
                    band = src.read(i)
                    
                    # 计算每个波段的高频能量
                    high_freq_energy, magnitude_spectrum, high_freq_magnitude = analyze_band_high_frequency_energy(band)
                    band_energies.append(high_freq_energy)

                    # # 可视化频谱和高频区域（可选）
                    # plt.figure(figsize=(12, 6))
                    # plt.suptitle(f"High Frequency Analysis for {filename} - Band {i}")
                    
                    # plt.subplot(1, 2, 1)
                    # plt.title("Magnitude Spectrum")
                    # plt.imshow(magnitude_spectrum, cmap='gray')
                    
                    # plt.subplot(1, 2, 2)
                    # plt.title("High Frequency Component")
                    # plt.imshow(np.log(high_freq_magnitude + 1), cmap='gray')
                    # plt.show()
                
                # 将每个波段的高频能量添加到字典中
                high_freq_energies[filename] = band_energies

    return high_freq_energies

ratio_list = []
ratios = np.arange(0.5, 1, 0.02)
for ratio in ratios:
    ratio_percentage = f"{1/ratio:.2f}"
    ratio_list.append(str(ratio_percentage))
print(ratio_list)

for ratio in ratio_list:
    # 示例用法
    folder_path = r'N:\项目数据\花江\语义分割\1-clip_img\建筑_1024_downsample_x' + ratio
    high_freq_energies = process_multiband_images_in_folder(folder_path)

    energies_mean = []
    # 输出所有图像的高频能量
    for filename, energies in high_freq_energies.items():
        for band_index, energy in enumerate(energies, start=1):
            #print(f"{filename} - Band {band_index}: High-frequency energy = {energy}")
            energies_mean.append(energy)
    print(np.mean(energies_mean))