#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
图片亮度批量参考调整
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
import rasterio
import numpy as np

def calculate_brightness(image_data):
    # 计算每个波段的平均值，并取所有波段的平均作为亮度
    return np.mean(image_data)

def adjust_brightness(source_image, target_brightness):
    # 计算源图像的亮度
    source_brightness = calculate_brightness(source_image)
    
    # 计算亮度调整系数
    brightness_factor = target_brightness / source_brightness
    
    # 调整亮度
    adjusted_image = source_image * brightness_factor
    return adjusted_image

def match_images_brightness(A_folder, B_folder, C_folder):
    # 确保C文件夹存在
    os.makedirs(C_folder, exist_ok=True)
    
    for file_name in os.listdir(A_folder):
        if file_name.endswith('.tif') and os.path.exists(os.path.join(B_folder, file_name)):
            A_img_path = os.path.join(A_folder, file_name)
            B_img_path = os.path.join(B_folder, file_name)
            C_img_path = os.path.join(C_folder, file_name)

            # 打开 A 文件夹和 B 文件夹中的图像
            with rasterio.open(A_img_path) as A_img, rasterio.open(B_img_path) as B_img:
                # 读取图像数据
                A_data = A_img.read()  # 多波段图像，形状为 (bands, height, width)
                B_data = B_img.read()
                
                # 计算目标图像的亮度
                target_brightness = calculate_brightness(B_data)
                
                # 调整 A 图像的亮度
                adjusted_data = adjust_brightness(A_data, target_brightness)
                
                # 将调整后的数据保存到 C 文件夹
                adjusted_data = adjusted_data.astype(A_data.dtype)  # 确保数据类型一致
                with rasterio.open(
                    C_img_path, 'w',
                    driver=A_img.driver,
                    width=A_img.width,
                    height=A_img.height,
                    count=A_img.count,
                    dtype=A_img.dtypes[0],
                    crs=A_img.crs,
                    transform=A_img.transform
                ) as dst:
                    dst.write(adjusted_data)

            print(f"Adjusted brightness for {file_name} and saved to {C_img_path}")

# 设置 A、B 和 C 文件夹路径
A_folder = r'E:\project_populus_GF2_and_UAV\0-clip_polygon_img/02-GF2-3-band'
B_folder = r'E:\project_populus_GF2_and_UAV\0-clip_polygon_img\14-UAV-321-resample-X2'
C_folder = r'E:\project_populus_GF2_and_UAV\0-clip_polygon_img/03-GF2-3-band-LA'

# 调用函数
match_images_brightness(A_folder, B_folder, C_folder)