# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型预测
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import numpy as np
import os
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import torch
from torch.autograd import Variable as V
import fnmatch
import sys
import math

from data import DataTrainInform

from networks.DLinknet import DLinkNet34, DLinkNet50, DLinkNet101
from networks.Unet import Unet
from networks.Dunet import Dunet
from networks.Deeplab_v3_plus import DeepLabv3_plus
from networks.FCN8S import FCN8S
from networks.DABNet import DABNet
from networks.Segformer import Segformer
from networks.RS_Segformer import RS_Segformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SolverFrame():
    def __init__(self, net):
        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

class Predict():
    def __init__(self, net, class_number, band_num):
        self.class_number = class_number
        self.img_mean = data_dict['mean']
        self.std = data_dict['std']
        self.net = net
        self.band_num = band_num
    
    def Predict_wHy(self, img_block, dst_ds, xoff, yoff):
        img_block = img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32)

        self.net.eval()

        for i in range(self.band_num):
            img_block[:, :, i] -= self.img_mean[i]
        img_block = img_block / self.std

        img_block = np.expand_dims(img_block, 0)
        img_block = img_block.transpose(0, 3, 1, 2)
        img_block = V(torch.Tensor(img_block).cuda())
        predict_out = self.net.forward(img_block).squeeze().cpu().data.numpy()

        predict_out = predict_out.transpose(1, 2, 0) # (h, w, c) -> (c, h, w)
        predict_result = np.argmax(predict_out, axis=2)
        dst_ds.GetRasterBand(1).WriteArray(predict_result, xoff, yoff)


    def Main(self, allpath, outpath, target_size=256, unify_read_img = False):  
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            dataset = gdal.Open(one_path)
            if dataset == None:
                print("failed to open img")
                sys.exit(1)
            img_width = dataset.RasterXSize
            img_height = dataset.RasterYSize

            '''新建输出tif'''
            d, n = os.path.split(one_path)

            projinfo = dataset.GetProjection() 
            geotransform = dataset.GetGeoTransform()

            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 数据格式
            name = n[:-4] + '_result' + '.tif'  # 输出文件名

            dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                                1, gdal.GDT_Byte)  # 创建一个新的文件
            dst_ds.SetGeoTransform(geotransform)  # 写入投影
            dst_ds.SetProjection(projinfo)  # 写入坐标

            
            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray() # 影像一次性读入内存
                #全局
                for i in tqdm(range(0, img_width-target_size, target_size)):
                    for j in range(0, img_height-target_size, target_size):
                        self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j)
                
                #下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

                #右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)

                #右下角
                self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size)
                dst_ds.FlushCache() # 缓存写入磁盘
            else:
                '''分块读取影像并预测'''
                # 全局
                for i in tqdm(range(0, math.floor(img_width/target_size-1)*target_size, target_size)):
                    for j in range(0, math.floor(img_height/target_size-1)*target_size, target_size):
                        img_block = dataset.ReadAsArray(i, j, target_size, target_size)
                        self.Predict_wHy(img_block.copy(), dst_ds, xoff=i, yoff=j)
                    dst_ds.FlushCache()
                    
                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    img_block = dataset.ReadAsArray(i, row_begin, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds, xoff=i, yoff=row_begin)
                dst_ds.FlushCache()
                
                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    img_block = dataset.ReadAsArray(col_begin, j, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds, xoff=col_begin, yoff=j)
                dst_ds.FlushCache()

                # 右下角
                img_block = dataset.ReadAsArray(img_width-target_size, img_height-target_size, target_size, target_size)
                self.Predict_wHy(img_block.copy(), dst_ds, img_width-target_size, img_height-target_size)
                dst_ds.FlushCache()

            print('预测耗费时间: %0.1f(s).' % (time.time() - t0))

if __name__ == '__main__':

    predictImgPath = r'E:\projict_UAV_yunnan\0-srimg_1026new\3channels' # 待预测影像的文件夹路径
    Img_type = '*.tif' # 待预测影像的类型
    trainListRoot = r'E:\projict_UAV_yunnan\2-trainlist\trainlist_1026_add_neg_1.txt' #与模型训练相同的trainlist
    numclass = 2 # 样本类别数
    model = DLinkNet34 #模型
    model_path = r'E:\projict_UAV_yunnan\3-weights\DLinkNet34-UAV_yunnan_yancao_1026_add_neg_1.th' # 模型文件完整路径
    output_path = r'E:\projict_UAV_yunnan\3-predict_result' # 输出的预测结果路径
    band_num = 3 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    target_size = 192 # 预测滑窗大小，应与训练集应一致
    unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True

    '''收集训练集信息'''
    dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
    data_dict = dataCollect.collectDataAndSave()

    print('data mean: ', data_dict['mean'])
    print('data std: ', data_dict['std'])

    '''初始化模型'''
    solver = SolverFrame(net = model(num_classes=numclass, band_num = band_num)) 
    solver.load(model_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    '''读取待预测影像'''
    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type)
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    if not listpic:
        print('listpic is none')
        exit(1)
    else:
        print(listpic)

    '''执行预测'''
    predict_instantiation = Predict(net=solver.net, class_number=numclass, band_num=band_num)
    predict_instantiation.Main(listpic, output_path, target_size, unify_read_img=unify_read_img)