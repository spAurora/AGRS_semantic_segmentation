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
            driver.CreateCopy(pngDir_path + '/'+ fileName[:-4] + ".png", ds)
            print("已生成：",pngDir_path +'/'+ fileName[:-4] + ".png")
 

tifDir_path = r"E:\project_UAV_GF2_2\4-clip_img_UAV_321_8bit_enhanced-X2"
pngDir_path = r"D:\github\ECDP\data\gupopulus_2\gupopulus_train_HR"
TIFToPNG(tifDir_path, pngDir_path)