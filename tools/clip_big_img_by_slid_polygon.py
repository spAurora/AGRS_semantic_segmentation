# -*- coding: utf-8 -*-

"""
代码简介
按照所需要的size的矩形批量对多波段整幅影像进行裁剪并输出多个矩形小块影像；
非矩形用nan补全；
参数(输入栅格文件存放文件夹路径，输出多个矩形栅格文件存放文件夹路径，使用波段数量，裁剪矩形大小)
~~~~~~~~~~~~~~~~
code by kunqi
Aerospace Information Research Institute, Chinese Academy of Sciences
kouwenqi22@mails.ucas.ac.cn
"""
import gdal
import numpy as np
from osgeo import gdal
import os
import pathlib
from rasterio.plot import show
import pandas as pd
import multiprocessing as mp
from itertools import repeat
import sys
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import Affine
from pathlib import Path
#裁剪(输入栅格文件存放文件夹路径，输出栅格文件存放文件夹路径，使用波段数量，裁剪大小)
def clipp(int_path,out_path,use_band,size):
    inList = [name for name in os.listdir(int_path)]
    for file in inList:
        if file.endswith('.tif'):
            print("待裁剪影像：",file)
            input_raster = os.path.join(int_path, file)
            # 输出文件的完整路径
            output_raster = os.path.join(out_path, file.strip(".tif"))
            if not os.path.exists(output_raster):
                os.makedirs(output_raster)
            in_ds = gdal.Open(input_raster)              # 读取要切的原图
            if in_ds is None:
                print("打开失败！")
            else:
                print("打开成功！")
                width = in_ds.RasterXSize  # 获取数据宽度
                height = in_ds.RasterYSize  # 获取数据高度
                outbandsize = in_ds.RasterCount  # 获取数据波段数
                im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
                im_proj = in_ds.GetProjection()  # 获取投影信息
                datatype = in_ds.GetRasterBand(1).DataType
                print("==================",file,"影像信息======================")
                print("width:",width,"height:",height,"outbandsize:",outbandsize)
                if outbandsize < use_band:
                    print("影像波段数小于所使用波段数！")
                else:
                    # 读取原图中所需的前ues_band个波段，并将前use_band个波段读入“data_all”
                    int_data_all = []
                    for inband in range(use_band):
                        int_band = in_ds.GetRasterBand(inband + 1)
                        int_data_all.append(int_band)
                    print('总共打开，并读取到内存的影像波段数目：{0}'.format(len(int_data_all)))

                    # 定义切图的大小（矩形框）
                    size = size
                    if size > width or size > height:
                        print("裁剪尺寸大于原始影像，请重新确定输入！")
                    else:
                        # 定义切图的起始点坐标
                        col_num = int(width / size)  # 宽度可以分成几块
                        row_num = int(height / size)  # 高度可以分成几块
                        if (width % size != 0):
                            col_num += 1
                        if (height % size != 0):
                            row_num += 1
                        num = 1  # 记录一共有多少块
                        print("row_num:%d   col_num:%d" % (row_num, col_num))
                        for i in range(col_num):  # 0-2
                            for j in range(row_num):  # 0-4
                                offset_x = j * size
                                offset_y = i * size
                                ## 从每个波段中切需要的矩形框内的数据
                                b_ysize = min(width - offset_y, size)
                                b_xsize = min(height - offset_x, size)
                                print(
                                    "width:%d     height:%d    offset_x:%d    offset_y:%d     b_xsize:%d     b_ysize:%d" % (
                                        width, height, offset_x, offset_y, b_xsize, b_ysize))
                                # print("\n")
                                out_data_all = []
                                for band in range(use_band):
                                    out_data_band = int_data_all[band].ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                                    out_data_band[np.where(out_data_band < 0)] = 0
                                    out_data_band[np.isnan(out_data_band)] = 0
                                    print("min", np.min(out_data_band), "max", np.max(out_data_band))
                                    out_data_all.append(out_data_band)
                                # print("out_data第{0}矩形已成功写入：{1}个波段".format(num, len(out_data_all)))
                                # 获取Tif的驱动，为创建切出来的图文件做准备
                                gtif_driver = gdal.GetDriverByName("GTiff")
                                file = output_raster + '\%04d.tif' % num
                                print("out_file", file)
                                num += 1
                                # 创建切出来的要存的文件
                                out_ds = gtif_driver.Create(file, size, size, outbandsize, datatype)
                                # print("create new tif file succeed")
                                # 获取原图的原点坐标信息
                                ori_transform = in_ds.GetGeoTransform()
                                # 读取原图仿射变换参数值
                                top_left_x = ori_transform[0]  # 左上角x坐标
                                w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
                                top_left_y = ori_transform[3]  # 左上角y坐标
                                n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

                                # 根据反射变换参数计算新图的原点坐标
                                top_left_x = top_left_x - offset_y * n_s_pixel_resolution
                                top_left_y = top_left_y - offset_x * w_e_pixel_resolution
                                print("top_left_x", top_left_x, "top_left_y", top_left_y)

                                # 将计算后的值组装为一个元组，以方便设置
                                dst_transform = (
                                    top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4],
                                    ori_transform[5])

                                # 设置裁剪出来图的原点坐标
                                out_ds.SetGeoTransform(dst_transform)

                                # 设置SRS属性（投影信息）
                                out_ds.SetProjection(in_ds.GetProjection())

                                # 写入目标文件
                                for w_band in range(use_band):
                                    out_ds.GetRasterBand(w_band + 1).WriteArray(out_data_all[w_band])

                                # 将缓存写入磁盘
                                out_ds.FlushCache()
                                print("=============已成功写入第{}个矩形===============".format(num))
                                del out_ds
            del in_ds
img_filepath = r"E:\DOM分类标签数据\1_打标签数据(未切片)\data"
out_filepath = r"E:\DOM分类标签数据\1-clip_img"
band_num = 3
rectangle_size = 4096
if len(sys.argv) > 2:
    img_filepath = sys.argv[1]
    out_filepath = sys.argv[2]
    band_num = sys.argv[3]
    rectangle_size = sys.argv[4]
clipp(img_filepath, out_filepath, int(band_num), int(rectangle_size))



