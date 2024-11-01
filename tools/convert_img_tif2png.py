# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
tif转png
~~~~~~~~~~~~~~~~
https://blog.csdn.net/ChaoChao66666/article/details/130214333
"""

from osgeo import gdal
import os
 
 
def TIFToPNG(tifDir_path, pngDir_path):
    for fileName in os.listdir(tifDir_path):
        if fileName[-4:] == ".tif":
            ds = gdal.Open(tifDir_path +'/' + fileName)
            driver = gdal.GetDriverByName('PNG')
            driver.CreateCopy(pngDir_path + '/'+ fileName[:-4] + "x2.png", ds)
 

tifDir_path = r"E:\project_UAV_GF2_2\3-clip_img_GF2_432_enhanced"
pngDir_path = r"D:\github\ECDP\data\gupopulus_2\gupopulus_train_LR"
TIFToPNG(tifDir_path, pngDir_path)