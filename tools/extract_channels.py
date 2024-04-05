#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
批量抽取影像指定通道
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import gdal
import ogr
import fnmatch
import os
import sys
import numpy as np
import math

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

os.environ['GDAL_DATA'] = r'C:\Users\75198\anaconda3\envs\learn\Lib\site-packages\osgeo\data\gdal' # To prevent ERROR4

img_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-clip_img\1-clip_img_haze_lv3'
output_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-clip_img\1-clip_img_haze_lv3_853'
save_channels = [8, 5, 3] # 顺序抽取的通道
# save_channels = [1] # 顺序抽取的通道

listpic = fnmatch.filter(os.listdir(img_path), '*.tif')

'''逐个读取影像'''
for img in listpic:
    img_full_path = img_path + '/' + img
    data, proj_temp, geotrans_temp = read_img(img_full_path)
    img_shape = data.shape
    data_temp = np.array(data[save_channels[0]-1, :, :]).reshape(1, img_shape[1], img_shape[2]) # (H, W) -> (1, H, W)

    '''抽取指定波段'''
    for i in range(1, len(save_channels)):
        temp = np.array(data[save_channels[i]-1, :, :]) # (H, W)
        data_temp = np.concatenate((data_temp, temp[None, :, :]))
        
    '''输出影像'''
    out_full_path = output_path + '/' + img
    write_img(out_full_path, proj_temp, geotrans_temp, data_temp)


