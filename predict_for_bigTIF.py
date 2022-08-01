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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

background = [0, 0, 0]
built_up = [255, 0, 0]
farmland = [0, 255, 0]
forest = [0, 255, 255]
meadow = [255, 255, 0]
water = [0, 0, 255]
COLOR_DICT = np.array([background, built_up, farmland, forest, meadow, water]) 

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
        dst_ds.FlushCache() # 缓存写入磁盘

    def main(self, allpath, outpath, target_size=256):  
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

            '''分块预测并写入输出影像'''
            # 全局
            for i in tqdm(range(0, math.floor(img_width/target_size-1)*target_size, target_size)):
                for j in range(0, math.floor(img_height/target_size-1)*target_size, target_size):
                    img_block = dataset.ReadAsArray(i, j, target_size, target_size)
                    self.Predict_wHy(img_block, dst_ds, xoff=i, yoff=j)
                
            # 下侧边缘
            row_begin = img_height - target_size
            for i in tqdm(range(0, img_width - target_size, target_size)):
                img_block = dataset.ReadAsArray(i, row_begin, target_size, target_size)
                self.Predict_wHy(img_block, dst_ds, xoff=i, yoff=row_begin)
            
            # 右侧边缘
            col_begin = img_width - target_size
            for j in tqdm(range(0, img_height - target_size, target_size)):
                img_block = dataset.ReadAsArray(col_begin, j, target_size, target_size)
                self.Predict_wHy(img_block, dst_ds, xoff=col_begin, yoff=j)

            # 右下角
            img_block = dataset.ReadAsArray(img_width-target_size, img_height-target_size, target_size, target_size)
            self.Predict_wHy(img_block, dst_ds, img_width-target_size, img_height-target_size)

            print('预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))

if __name__ == '__main__':

    predictImgPath = r'G:\manas_class\project_manas\0-src_img' # 待预测影像的文件夹路径
    Img_type = '*.dat' # 待预测影像的类型
    trainListRoot = r'G:\manas_class\project_manas\water\2-trainlist\trainlist_0727_balance_test.txt' #与模型训练相同的trainlist
    numclass = 2 # 样本类别数
    model = Dunet #模型
    model_path = r'G:\manas_class\project_manas\water\3-weights\Dunet-manans_water_balance_0728test.th' # 模型文件完整路径
    output_path = r'C:\Users\75198\Desktop\check' # 输出的预测结果路径
    band_num = 4 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    overlap_rate = 0
    target_size = 256 # 预测滑窗大小，应与训练集应一致

    model_name = model.__class__.__name__
    print(model_name)

    dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
    data_dict = dataCollect.collectDataAndSave()

    print('data mean: ', data_dict['mean'])
    print('data std: ', data_dict['std'])

    solver = SolverFrame(net = model(num_classes=numclass, band_num = band_num)) 
    solver.load(model_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type)
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    if not listpic:
        print('listpic is none')
        exit(1)
    else:
        print(listpic)

    predict_instantiation = Predict(net=solver.net, class_number=numclass, band_num=band_num)
    predict_instantiation.main(listpic, output_path, target_size)





