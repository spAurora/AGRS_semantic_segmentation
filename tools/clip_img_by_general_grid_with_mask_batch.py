# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
带有掩膜的均匀格网裁剪
与序列掩膜预测的掩膜生成规则一致
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
import mmap


if_vismem = True

FQ_NAME_LIST = ['HOTAN', 'KASHGAR','KLIYA','QARQAN','TARIM']


for fq_name in FQ_NAME_LIST:
    # 输入影像路径
    input_image_dir = r'K:\PROJECT_GLOBAL_POPULUS_DATA_03\FQ-XJ_' + fq_name +'0316\IMAGE-FUSE'
    # 输出小块保存路径
    output_dir = r'K:\PROJECT_GLOBAL_POPULUS_DATA_03\FQ-XJ_' + fq_name +'0316\CLIP_TILE'
    # 掩膜路径
    mask_path = r'K:\PROJECT_GLOBAL_POPULUS_DATA_03\FQ-XJ_' + fq_name +'0316\MASK' # mask路径 路径下需要有*.npz掩膜（./tools/generate_mask_by_moasic_line.py生成）


    os.makedirs(output_dir, exist_ok=True)

    # 裁剪小块数量和尺寸
    num_tiles = 2500
    tile_size = 256  # 小块大小（宽和高)

    process_block = 1 # 分块处理

    total_tiles = 0
    img_cnt = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    listpic = fnmatch.filter(os.listdir(input_image_dir), '*.img')

    for img in listpic:

        img_cnt+=1  

        mask_full_path = mask_path + '/' + img[:-4] + '.npz'
        if os.path.exists(mask_full_path):
            m_data = np.load(mask_path + '/' + img[:-4] + '.npz')
            mask = m_data['mask']
        else:
            print('Mask file does not exist: ' + mask_full_path)
            continue

        input_image_path = input_image_dir + '/' + img

        if if_vismem:
            # 获取影像文件名前缀（不包括扩展名）
            filename_prefix = os.path.splitext(os.path.basename(input_image_path))[0]

            input_dir = os.path.dirname(input_image_path)
            # 搜索配套文件（除主影像文件外其他相同前缀的文件）
            auxiliary_files = [
                f for f in os.listdir(input_dir)
                if f.startswith(filename_prefix) and f != os.path.basename(input_image_path)
            ]

            # 将主影像文件映射到内存
            with open(input_image_path, "rb") as f:
                mmapped_main_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                gdal.FileFromMemBuffer("/vsimem/" + os.path.basename(input_image_path), mmapped_main_file[:])

            # 将配套文件映射到内存
            for aux_file in auxiliary_files:
                aux_path = os.path.join(input_dir, aux_file)
                with open(aux_path, "rb") as f_aux:
                    mmapped_aux_file = mmap.mmap(f_aux.fileno(), 0, access=mmap.ACCESS_READ)
                    gdal.FileFromMemBuffer(f"/vsimem/{aux_file}", mmapped_aux_file[:])

            dataset = gdal.Open("/vsimem/" + os.path.basename(input_image_path), gdal.GA_ReadOnly) # GDAL打开待切分影像
        else:
            dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly) # GDAL打开待切分影像

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

                    # 裁剪小块
                    tile = data[:, y:y + tile_size, x:x + tile_size]
                    mask_patch = mask[y:y + tile_size, x:x + tile_size]
                    
                    # 跳过全为 0 的小块(包括图像和掩膜)
                    if np.all(tile == 0) or not np.all(mask_patch == 1):
                        continue

                    actual_tiles += 1
                    total_tiles += 1

                    # 构造输出文件名
                    output_tile_path = os.path.join(output_dir, fq_name + '_' + str(img_cnt) + f"_{actual_tiles + 1:08d}.tif")
                    
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
                    # 清理内存文件
        if if_vismem:
            gdal.Unlink("/vsimem/" + os.path.basename(input_image_path))
            for aux_file in auxiliary_files:
                gdal.Unlink(f"/vsimem/{aux_file}")