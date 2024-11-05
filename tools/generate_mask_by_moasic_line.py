# -*- coding: utf-8 -*-

"""
基于接缝线生成对应的mask掩膜
强烈建议使用GDAL3.X版本，可极大提高生成速度
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
import sys
import fnmatch
import numpy as np
from osgeo import gdal, ogr

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

    return im_data, im_proj, im_geotrans, im_width, im_height


image_path = r'E:\hami\image'
image_type = '*.img'
mosaic_line_full_path = r'E:\hami\mosaic_line\Mosaic line edited.shp' # 镶嵌线文件绝对路径，注意镶嵌线文件其实是个polygon
ouput_path = r'E:\hami\mask'

if not os.path.exists(ouput_path):
    os.mkdir(ouput_path)

# 读取镶嵌线文件
vector = ogr.Open(mosaic_line_full_path)
if vector == None:
    print('读取镶嵌线文件失败')
layer = vector.GetLayer()

image_list = fnmatch.filter(os.listdir(image_path), image_type)

for image in image_list:

    print('processing ' + image)

    image_full_path = image_path + '/' + image
    output_full_path = ouput_path + '/' + image[:-4] + '.tif'

    im_data, im_proj, im_geotrans, im_width, im_height = read_img(image_full_path)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_full_path, im_width, im_height, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(im_geotrans)
    ds.SetProjection(im_proj)

    layer.SetAttributeFilter("LayerName = '{}'".format(image[:-4])) # 文件名必须与LayerName完全对应

    gdal.RasterizeLayer(ds, [1], layer, burn_values=[1]) # 栅格化

    raster_data = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    np.savez_compressed(output_full_path[:-4] + '.npz', mask=raster_data)

    ds = None
    
    os.remove(output_full_path) # 如果要debug可以注释掉这句输出掩膜影像

vector.Destroy()

print('done')
    
