#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
批量抽取影像指定通道（并行优化版）
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

from osgeo import gdal
from osgeo import ogr
import fnmatch
import os
import sys
import numpy as np
import math
import time
from multiprocessing import Pool, cpu_count
import functools

def write_img(out_path, im_proj, im_geotrans, im_data):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # identify data type 
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # calculate number of bands
    if len(im_data.shape) > 2:  
        im_bands, im_height, im_width = im_data.shape
    else:  
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data.squeeze())
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset

def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    del im_dataset

    return im_data, im_proj, im_geotrans

def process_single_image(img, img_path, output_path, T_value):
    """处理单个图像的函数，用于并行处理"""
    try:
        img_full_path = os.path.join(img_path, img)
        data, proj_temp, geotrans_temp = read_img(img_full_path)
        img_shape = data.shape
                
        data_temp = np.where(data < T_value, 0, 255)
        
        # 输出影像
        out_full_path = os.path.join(output_path, img)
        write_img(out_full_path, proj_temp, geotrans_temp, data_temp)
        
        return f"Processed {img} successfully"
    except Exception as e:
        return f"Error processing {img}: {str(e)}"

def main():
    # os.environ['GDAL_DATA'] = r'C:\Users\75198\anaconda3\envs\learn\Lib\site-packages\osgeo\data\gdal' # To prevent ERROR4

    img_path = r'D:\github_respository\Populus_SR_GF2_UAV\data\gupopulus_2\gupopulus_train_LR_shadow\BAND_NIR'
    output_path = r'D:\github_respository\Populus_SR_GF2_UAV\data\gupopulus_2\gupopulus_train_LR_shadow\MASK'
    
    T_value = 50 # 二值化阈值  
    

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 获取文件列表
    listpic = fnmatch.filter(os.listdir(img_path), '*.png')
    
    # 确定使用的进程数（使用CPU核心数-1，留一个核心给系统）
    num_processes = max(1, cpu_count() - 1)
    
    print(f"Starting processing {len(listpic)} images using {num_processes} processes...")
    start_time = time.time()
    
    # 使用functools.partial创建带有固定参数的函数
    process_func = functools.partial(
        process_single_image,
        img_path=img_path,
        output_path=output_path,
        T_value=T_value
    )
    
    # 创建进程池并并行处理
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_func, listpic)
    
    # 打印处理结果
    for result in results:
        print(result)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()