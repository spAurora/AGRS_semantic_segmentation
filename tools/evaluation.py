#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
精度评定
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import os
from statistics import mean
import sys
import fnmatch
import numpy as np
import gdal
import ogr


# os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

ground_truth_path = r'E:\project_UAV\1-raster_label' # 存储待评定真值标签的文件夹
predict_shp_path = r'E:\project_UAV\1-artificial_shp' # 存储预测矢量的文件

gt_list = fnmatch.filter(os.listdir(ground_truth_path), '*.tif') # 过滤出所有tif文件

TP = []
TN = []
FP = []
FN = []

PA = []
UA = []
OA = []
F1 = []
MIoU = []

'''遍历真值数据'''
for gt_file in gt_list:
    gt_file = os.path.join(ground_truth_path + '/' + gt_file)

    print(gt_file)
    
    '''读取真值数据'''
    image_gt = gdal.Open(gt_file)
    geotransform = image_gt.GetGeoTransform()             
    ref = image_gt.GetProjection()
    x_res = image_gt.RasterXSize
    y_res = image_gt.RasterYSize
    data_gt = image_gt.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    data_gt[np.where(data_gt==0)] = 0 # 真值的背景值
    data_gt[np.where(data_gt>0)] = 1 # 真值的前景值

    '''读取预测结果'''
    vector = ogr.Open(predict_shp_path)
    if vector == None:
        print('第二次shp文件失败')
    layer = vector.GetLayer()
    targetDataset = gdal.GetDriverByName('GTiff').Create('temp.tif', x_res, y_res, 3, gdal.GDT_Byte)
    targetDataset.SetGeoTransform(image_gt.GetGeoTransform())
    targetDataset.SetProjection(image_gt.GetProjection())
    band = targetDataset.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1, 2, 3], layer, )
    targetDataset = None

    image_predict = gdal.Open('temp.tif')
    data_predict = image_predict.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    data_predict[np.where(data_predict==0)] = 0 # 预测结果的背景值
    data_predict[np.where(data_predict>0)] = 1 # 预测结果的前景值

    '''统计TP FP TN FN'''
    TP_tmp = 0
    FP_tmp = 0
    TN_tmp = 0
    FN_tmp = 0
    for i in range(x_res):
        for j in range(y_res):
            if data_gt[i,j] == 1 and data_predict[i,j] == 1:
                TP_tmp += 1
            elif data_gt[i,j] == 0 and data_predict[i,j] == 0:
                TN_tmp += 1
            elif data_gt[i,j] == 0 and data_predict[i,j] == 1:
                FP_tmp += 1
            elif data_gt[i,j] == 1 and data_predict[i,j] == 0:
                FN_tmp += 1
            else:
                print('stat TP TN FP FN wrong: data_gt[i,j]=' + str(data_gt[i,j]) + ', data_predict[i,j]=' + str(data_predict[i,j]))
                sys.exit(1)

    '''计算精度评定指标'''
    OA_tmp = (TP_tmp+TN_tmp)/(TP_tmp+TN_tmp+FP_tmp+FN_tmp)
    PA_tmp = TP_tmp/(TP_tmp+FP_tmp)
    UA_tmp = TP_tmp/(TP_tmp+FN_tmp)
    F1_tmp = 2*(PA_tmp*UA_tmp)/(PA_tmp+UA_tmp)
    MIoU_tmp = TP_tmp/(FN_tmp+FP_tmp+TP_tmp)

    print(TP_tmp, TN_tmp, FP_tmp, FN_tmp, PA_tmp, UA_tmp, OA_tmp, F1_tmp, MIoU_tmp)

    TP.append(TP_tmp)
    TN.append(TN_tmp)
    FP.append(FP_tmp)
    FN.append(FN_tmp)
    OA.append(OA_tmp)
    PA.append(PA_tmp)
    UA.append(UA_tmp)
    F1.append(F1_tmp)
    MIoU.append(MIoU_tmp)
    
    '''释放指针'''
    image_predict = None


print('AVERAGE: PA = %.3f' % mean(PA), 'UA = %.3f' % mean(UA), 'OA = %.3f' % mean(OA), 'F1 = %.3f' % mean(F1), 'MIoU = %.3f' % mean(MIoU))

os.remove('temp.tif') # 清理缓存文件
