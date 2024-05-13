#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
删除矢量化结果中的背景
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

os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

shp_img_path = r'E:\projict_UAV_yunnan\4-predict_result_shp'

listpic = fnmatch.filter(os.listdir(shp_img_path), '*.shp')

for shp in tqdm(listpic):
    shp_full_path = shp_img_path + '/' + shp

    ogr.RegisterAll()# 注册所有的驱动

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_dataset = ogr.Open(shp_full_path, 1) # 0只读模式，1读写模式
    if shp_full_path is None:
        print('Failed to open shp')

    ly = shp_dataset.GetLayer()

    '''删除矢量化结果中gridcode=0的要素'''
    feature = ly.GetNextFeature()
    while feature is not None:
        gridcode = feature.GetField('gridcode')
        if gridcode == 0:
            delID = feature.GetFID()
            ly.DeleteFeature(int(delID))
        feature = ly.GetNextFeature()
    ly.ResetReading() #重置
    del shp_dataset