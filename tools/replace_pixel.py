# -*- coding: utf-8 -*-

"""
替换图片像素值
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
Ghent University
Haoyu.Wang@ugent.be
"""
from PIL import Image

image_path = r'E:\paper_lishuo_new\fig-全域地块提取示意图\wgh.tif'  # 替换为您的图像路径
output_path = r'E:\paper_lishuo_new\fig-全域地块提取示意图\wgh_replace.tif'

# 打开RGB图像
img = Image.open(image_path)

# 将图像转换为RGB模式，确保图像处理时不受模式限制
img = img.convert("RGB")

pixel_before = [(31,120,180),(204, 88, 202),(84,187,232),(242,214,143),(153,101,127),(45,64,128),(255,0,197)]
pixel_after = [(51,160,44),(51, 160, 44),(31, 120, 180),(251,154,153),(227,26,28),(166,206,227),(255,127,0)]

# 获取像素数据
pixels = img.load()

# 遍历图像的每个像素
for i in range(len(pixel_before)):
    for y in range(img.height):
        for x in range(img.width):
            # 替换
            if pixels[x, y] == pixel_before[i]:
                pixels[x, y] = pixel_after[i]

# 保存图像为无损压缩的TIFF格式
img.save(output_path, format='TIFF', compression='tiff_lzw')  # 使用LZW无损压缩保存TIFF文件