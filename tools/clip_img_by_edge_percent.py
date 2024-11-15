# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
基于到边缘的比例裁剪图片
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
from PIL import Image

def crop_image(input_path, output_path, top_percent=30, bottom_percent=20):
    # 打开图像
    with Image.open(input_path) as img:
        # 获取图像尺寸
        width, height = img.size
        
        # 计算裁剪的尺寸
        top_crop = height * (top_percent / 100)
        bottom_crop = height * (bottom_percent / 100)
        new_height = height - top_crop - bottom_crop
        
        # 裁剪图像
        cropped_img = img.crop((0, top_crop, width, height - bottom_crop))
        
        # 保存图像，不压缩
        cropped_img.save(output_path, 'PNG', compress_level=0)

# 使用示例
input_image_path = 'path_to_your_image.jpg'  # 替换为你的图片路径
output_image_path = 'path_to_save_cropped_image.png'  # 替换为你想要保存的路径
crop_image(input_image_path, output_image_path)