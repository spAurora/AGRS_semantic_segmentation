# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型测试
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 放在import torch前，防止报OMP ERROR #15
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import torch
from torch.autograd import Variable
import fnmatch
import sys
import math

from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def precision_recall(cm):
    # 计算每个类的TP, FP, FN
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    # 计算每个类的精确度和召回率
    n = cm.shape[0] # 类别数
    precision = np.zeros(n)
    recall = np.zeros(n)

    for i in range(n):
        if (tp[i] + fp[i] == 0):
            precision[i] = 0
        else:
            precision[i] = tp[i] / (tp[i] + fp[i])

        if (tp[i] + fn[i] == 0):
            recall[i] = 0
        else:
            recall[i] = tp[i] / (tp[i] + fn[i])

    return precision, recall

def macro_average(precision, recall):
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1_score = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
    return macro_precision, macro_recall, macro_f1_score

def cal_cm_score(cm):
    precision, recall = precision_recall(cm)
    precision = np.nan_to_num(precision, nan=0)
    recall = np.nan_to_num(recall, nan=0)
    macro_precision, macro_recall, macro_f1_score = macro_average(precision, recall) # 计算宏平均精确度，宏平均召回率，宏平均 F1 分数
    return macro_precision, macro_recall, macro_f1_score

class TestFrame():
    def __init__(self, net, data_dict, band_num, if_norm_label, ignore_bandnum=0):
        self.img_mean = data_dict['mean'] # 数据集均值
        self.std = data_dict['std'] # 数据集方差
        self.net = net # 模型
        self.band_num = band_num # 影像波段数
        self.if_norm_label = if_norm_label
        self.ignore_bandnum = ignore_bandnum
    
    def Predict_wHy(self, img_block, dst_ds, xoff, yoff):
        img_block = img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32) # 数据类型转换

        self.net.eval() # 启动预测模式

        for i in range(self.band_num-self.ignore_bandnum): # 数据标准化
            img_block[:, :, i] -= self.img_mean[i]
            if self.std[i] != 0:
                img_block[:, :, i] = img_block[:, :, i] / self.std[i]

        img_block = np.expand_dims(img_block, 0) # 扩展数据维度 (h, w, c) -> (b, h, w, c) 
        img_block = img_block.transpose(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)
        img_block = Variable(torch.Tensor(img_block).cuda()) # Variable容器装载
        predict_out = self.net.forward(img_block).squeeze().cpu().data.numpy() # 模型预测；删除b维度；转换为numpy

        predict_out = predict_out.transpose(1, 2, 0) # (c, h, w) -> (h, w, c)
        predict_result = np.argmax(predict_out, axis=2) # 返回第三维度最大值的下标
        dst_ds.GetRasterBand(1).WriteArray(predict_result, xoff, yoff) # 预测结果写入gdal_dataset

    def Test_Main(self, target_size, img_type, test_img_path, test_label_path, test_output_path):
        '''读取待测试影像'''
        listpic = fnmatch.filter(os.listdir(test_img_path), img_type) # 过滤对应文件类型
        list_pic_full_path = []
        list_pre_full_path = []
        for i in range(len(listpic)):
            list_pic_full_path.append(os.path.join(test_img_path + '/' + listpic[i]))
            list_pre_full_path.append(os.path.join(test_output_path + '/' + listpic[i][:-4] + '_test.tif'))
        
        if not listpic:
            print('test pic is none')
            return -1
        
        p = 0
        r = 0
        f = 0
        cnt = 0
        for i in range(len(listpic)):
            one_path = list_pic_full_path[i]
            pre_full_path = list_pre_full_path[i]
            dataset = gdal.Open(one_path) # GDAL打开待测试影像
            if dataset == None:
                print("failed to open img")
                sys.exit(1)
            img_width = dataset.RasterXSize # 读取影像宽度
            img_height = dataset.RasterYSize # 读取影像高度

            '''新建输出tif'''
            projinfo = dataset.GetProjection() # 获取原始影像投影
            geotransform = dataset.GetGeoTransform() # 获取原始影像地理坐标

            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 数据格式

            dst_ds = driver.Create(pre_full_path, dataset.RasterXSize, dataset.RasterYSize,
                                1, gdal.GDT_Byte)  # 创建预测结果写入文件
            dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
            dst_ds.SetProjection(projinfo)  # 写入投影

            '''集中读取待测试影像并预测'''
            img_block = dataset.ReadAsArray() # 影像一次性读入内存
            # 全局整体
            for i in range(0, img_width-target_size, target_size):
                for j in range(0, img_height-target_size, target_size):
                    self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j)
            
            # 下侧边缘
            row_begin = img_height - target_size
            for i in range(0, img_width - target_size, target_size):
                self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

            # 右侧边缘
            col_begin = img_width - target_size
            for j in range(0, img_height - target_size, target_size):
                self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)

            # 右下角
            self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size)
            
            dst_ds.FlushCache() # 全部预测完毕后统一写入磁盘

            dst_ds = None

            '''读取temp预测图像和对应的真值图像'''
            # gt_full_path = test_label_path + '/' + one_path[:-4] + '_label.tif'
            gt_full_path = test_label_path + '/' + listpic[cnt]

            if not listpic:
                print('label pic is none')
                print('check: ' + gt_full_path)
                return -1
            
            input_pre = gdal.Open(pre_full_path)
            input_gt = gdal.Open(gt_full_path)

            # 定义裁剪窗口大小
            if input_pre.RasterXSize != input_gt.RasterXSize:
                print('predict img and label pic not match')
                return -1
            else:
                win_size = input_pre.RasterXSize

            im_data_pre = input_pre.ReadAsArray(0, 0, win_size, win_size)  # 读取预测结果数据
            im_data_true = input_gt.ReadAsArray(0, 0, win_size, win_size) # 读取真值标签数据

            if self.if_norm_label is True:
                im_data_true = im_data_true/255

            im_data_pre = list(im_data_pre.reshape(-1)) # 展平为一维
            im_data_true = list(im_data_true.reshape(-1).astype(np.uint8)) # 展平为一维

            #print(type(im_data_pre[0]), type(im_data_true[0]))

            '''精度评定'''
            cm = confusion_matrix(im_data_true, im_data_pre, normalize='true') # 首先计算归一化混淆矩阵
            macro_precision, macro_recall, macro_f1_score = cal_cm_score(cm)
            p += macro_precision
            r += macro_recall
            f += macro_f1_score

            '''一次循环的后处理'''
            input_gt = None
            input_pre = None
            # os.remove(pre_full_path)
            cnt += 1
        
        '''循环完毕后返回各项指标'''
        return p/len(listpic), r/len(listpic), f/len(listpic)


def GetTestIndicator(net, data_dict, target_size, band_num, img_type, test_img_path, test_label_path, if_norm_label, test_output_path,ignore_bandnum):        
    '''执行预测'''
    test_instantiation = TestFrame(net=net, data_dict=data_dict, band_num=band_num, if_norm_label=if_norm_label,ignore_bandnum=ignore_bandnum) # 初始化预测
    p, r, f = test_instantiation.Test_Main(target_size=target_size, img_type=img_type, test_img_path=test_img_path, test_label_path=test_label_path, test_output_path=test_output_path)

    return p, r, f