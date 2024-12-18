# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
均匀格网裁剪
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

from osgeo import gdal
import os
import math
import fnmatch
import numpy as np

# 输入影像路径
input_image_dir = r"E:\Nanjiang2_mosaic\NJ2_mosaic\DOM\mosaic"

# 输出小块保存路径
output_dir = r"I:\PROJECT_GLOBAL_POPULUS_MAE_02\CLIP_IMG"
os.makedirs(output_dir, exist_ok=True)

# 裁剪小块数量和尺寸
num_tiles = 50000
tile_size = 256  # 小块大小（宽和高)

process_block = 10 # 分块处理

total_tiles = 0

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

listpic = fnmatch.filter(os.listdir(input_image_dir), '*.img')

for img in listpic:

    input_image_path = input_image_dir + '/' + img
    # 打开影像
    dataset = gdal.Open(input_image_path)
    if not dataset:
        raise FileNotFoundError(f"Cannot open the file: {input_image_path}")

    # 读取影像信息
    cols = dataset.RasterXSize  # 宽度
    rows = dataset.RasterYSize  # 高度
    bands = dataset.RasterCount  # 波段数量
    new_rows = int(rows/process_block)
    new_num_tiles = num_tiles/process_block

    for j in range(0, process_block):
        # 读取影像数据
        row_begin = new_rows * j
        data = dataset.ReadAsArray(0, row_begin, cols, new_rows)

        # 检查影像尺寸是否满足裁剪要求
        if cols < tile_size or new_rows < tile_size:
            raise ValueError("Image dimensions are smaller than the tile size.")

        # 计算网格划分
        grid_x = int(math.sqrt(new_num_tiles * (cols / new_rows)))  # 水平方向网格数
        grid_y = int(math.sqrt(new_num_tiles * (new_rows / cols)))  # 垂直方向网格数

        # 计算网格步长
        step_x = max(1, cols // grid_x)
        step_y = max(1, new_rows // grid_y)

        # 确保网格不超出实际需要的小块数
        actual_tiles = 0

        # 按网格裁剪并保存小块
        for y in range(0, new_rows - tile_size + 1, step_y):
            for x in range(0, cols - tile_size + 1, step_x):
                if actual_tiles >= new_num_tiles:
                    break
                
                actual_tiles += 1
                total_tiles += 1

                # 裁剪小块
                tile = data[:, y:y + tile_size, x:x + tile_size]
                
                # 跳过全为 0 的小块
                if np.all(tile == 0):
                    continue

                # 构造输出文件名
                output_tile_path = os.path.join(output_dir, f"tile_{total_tiles + 1:08d}.tif")
                
                # 创建小块影像
                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(output_tile_path, tile_size, tile_size, bands, gdal.GDT_Byte, options=["COMPRESS=LZW"])
                
                # 写入波段数据
                for band in range(bands):
                    out_ds.GetRasterBand(band + 1).WriteArray(tile[band])
                
                # 释放资源
                out_ds.FlushCache()
                out_ds = None
            
            if actual_tiles >= new_num_tiles:
                break

        print(f"Saved {actual_tiles} tiles to {output_dir}.")