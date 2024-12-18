#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
设置图像的nodata值
这个写法不好，需要遍历所有波段；张驰说用rasterio效率可以更高
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
from pathlib import Path
import gdal
import os
import ogr
import osr
import sys
import math
from osgeo.ogr import Geometry, Layer
from tqdm import tqdm
import numpy as np
import fnmatch
import copy

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
        new_dataset.GetRasterBand(1).WriteArray(im_data)
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

    return im_dataset, im_data, im_proj, im_geotrans, im_width, im_height



input_folder = r'E:\project_UAV_GF2_2\1-clip_img_UAV'  # 输入文件夹路径
output_folder = r'E:\project_UAV_GF2_2\1-clip_img_UAV_remove_nodata'  # 输出文件夹路径

img_type = '*.tif'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

listpic = fnmatch.filter(os.listdir(input_folder), img_type)

'''逐个读取影像'''
for img in listpic:
    img_full_path = input_folder + '/' + img
    im_dataset, data, proj_temp, geotrans_temp, width, height = read_img(img_full_path)
    img_shape = data.shape
    for i in range(img_shape[0]): # 读取每个波段
        nodatavalue = im_dataset.GetRasterBand(1).GetNoDataValue()
        print(nodatavalue)
        output_data = im_dataset.ReadAsArray(0, 0, width, height)
        output_data[np.where(output_data == nodatavalue)] = 0 # 修改后的nodatavalue

    output_full_path = output_folder + '/' + img
    write_img(output_full_path, proj_temp, geotrans_temp, output_data)