# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型预测
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
from multiprocessing.spawn import import_main_path
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
from networks.DE_Segformer import DE_Segformer


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
    
    def Predict_wHy(self, img_block, dst_ds, xoff, yoff, target_size, weight_block, num_classes):
        img_block = img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32)

        self.net.eval()

        for i in range(self.band_num):
            img_block[:, :, i] -= self.img_mean[i]
        img_block = img_block / self.std

        img_block = np.expand_dims(img_block, 0)
        img_block = img_block.transpose(0, 3, 1, 2) # (b, c, h, w) -> (b, h, w, c)
        img_block = V(torch.Tensor(img_block).cuda())
        predict_out = self.net.forward(img_block).squeeze().cpu().data.numpy() # output Tensor (c, h, w)

        predict_current = np.array(dst_ds.ReadAsArray(xoff, yoff, target_size, target_size), dtype=float) # (c, h, w)
        predict_current = (predict_current*weight_block + predict_out) / (weight_block + 1)
        weight_block += 1

        #predict_result = np.argmax(predict_out, axis=2)
        for i in range(num_classes):
            dst_ds.GetRasterBand(i+1).WriteArray(predict_current[i,:,:], xoff, yoff) # 数据写入内存


    def Main(self, allpath, outpath, target_size=256, unify_read_img = False, overlap_rate = 0, num_classes = 3):  
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

            dst_ds = driver.Create(os.path.join(outpath, name+'_temp'), dataset.RasterXSize, dataset.RasterYSize,
                                num_classes, gdal.GDT_Float32)  # 存储各类别的概率
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projinfo)

            rst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                                1, gdal.GDT_Byte) # 存储最终类别
            rst_ds.SetGeoTransform(geotransform)
            rst_ds.SetProjection(projinfo)

            weights = np.zeros((img_height, img_width), dtype=np.float32)
            step = int(target_size * (1-overlap_rate)) # overlap_rate控制步长

            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray() # 影像一次性读入内存
                #全局
                for i in tqdm(range(0, img_width-target_size, step)):
                    for j in range(0, img_height-target_size, step):
                        self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)
                
                #下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, step)):
                    self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)

                #右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, step)):
                    self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)

                #右下角
                self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)

            else:
                '''分块读取影像并预测'''
                # 全局
                for i in tqdm(range(0, math.floor(img_width/target_size-1)*target_size, step)):
                    for j in range(0, math.floor(img_height/target_size-1)*target_size, step):
                        img_block = dataset.ReadAsArray(i, j, target_size, target_size)
                        self.Predict_wHy(img_block.copy(), dst_ds, xoff=i, yoff=j, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)
                    
                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, step)):
                    img_block = dataset.ReadAsArray(i, row_begin, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds, xoff=i, yoff=row_begin, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)
                
                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, step)):
                    img_block = dataset.ReadAsArray(col_begin, j, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds, xoff=col_begin, yoff=j, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)

                # 右下角
                img_block = dataset.ReadAsArray(img_width-target_size, img_height-target_size, target_size, target_size)
                self.Predict_wHy(img_block.copy(), dst_ds, img_width-target_size, img_height-target_size, target_size=target_size, weight_block= weights[j:j+target_size, i:i+target_size], num_classes=num_classes)

            '''统一argmax'''
            predict_result = dst_ds.ReadAsArray()
            class_result = np.argmax(predict_result, axis=0)
            rst_ds.GetRasterBand(1).WriteArray(class_result*255, 0, 0)
            rst_ds.FlushCache() #写入硬盘
            del rst_ds
            del dst_ds
            del dataset
            print('预测耗费时间: %0.1f(s).' % (time.time() - t0))

if __name__ == '__main__':

    predictImgPath = r'D:\0-zhuanhuan2\srimg' # 待预测影像的文件夹路径
    Img_type = '*.tif' # 待预测影像的类型
    trainListRoot = r'E:\project_UAV\2-trainlist\trainlist_0910_1.txt' #与模型训练相同的trainlist
    numclass = 2 # 样本类别数
    model = DE_Segformer #模型
    model_path = r'E:\project_UAV\3-weights\DE_Segformer_N-UAV_building_1008.th' # 模型文件完整路径
    output_path = r'D:\0-zhuanhuan2\predict_result\DE_Segformer_N' # 输出的预测结果路径
    band_num = 3 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    target_size = 256 # 预测滑窗大小，应与训练集应一致
    unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True
    overlap_rate = 0 # 预测滑窗间的重叠度 取值在0-1之间

    '''收集训练集信息'''
    dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
    data_dict = dataCollect.collectDataAndSave()

    '''手动设置data_dict'''
    #data_dict = {}
    #data_dict['mean'] = [125.304955, 127.38818,  114.94185]
    #data_dict['std'] = [40.3933, 35.64181, 37.925995]
    #data_dict['classWeights'] = np.ones(2, dtype=np.float32)
    #data_dict['img_shape'] = (256, 256, 3)

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
    predict_instantiation.Main(listpic, output_path, target_size, unify_read_img=unify_read_img, overlap_rate = overlap_rate, num_classes = numclass)





