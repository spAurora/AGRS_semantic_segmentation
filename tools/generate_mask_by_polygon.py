# -*- coding: utf-8 -*-

"""
基于矢量多边形生成mask掩膜
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
    # im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    im_data = 0
    del im_dataset

    return im_data, im_proj, im_geotrans, im_width, im_height


image_path = r''
image_type = '*.dat'
mask_shp_path = r'' # 掩膜shp的文件夹路径，注意掩膜文件名和影像名要完全一致
ouput_path = r''

if not os.path.exists(ouput_path):
    os.mkdir(ouput_path)



image_list = fnmatch.filter(os.listdir(image_path), image_type)

for image in image_list:

    print('processing ' + image)

    image_full_path = image_path + '/' + image
    mask_shp_full_path = mask_shp_path + '/' + image[:-4] + '.shp'
    output_full_path = ouput_path + '/' + image[:-4] + '.tif'

    im_data, im_proj, im_geotrans, im_width, im_height = read_img(image_full_path)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_full_path, im_width, im_height, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(im_geotrans)
    ds.SetProjection(im_proj)

    # 读取polygon掩膜文件
    vector = ogr.Open(mask_shp_full_path)
    if vector == None:
        print('读取掩膜矢量文件失败')
    layer = vector.GetLayer()

    # 检查图层要素数目
    print('图层要素数： ', layer.GetFeatureCount())

    gdal.RasterizeLayer(ds, [1], layer, burn_values=[1]) # 栅格化

    raster_data = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    np.savez_compressed(output_full_path[:-4] + '.npz', mask=raster_data)

    ds = None
    os.remove(output_full_path) # 如果要debug可以注释掉这句输出掩膜影像

    vector.Destroy()

print('done')