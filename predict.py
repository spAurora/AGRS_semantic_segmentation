# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型预测
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

from data import DataTrainInform

from networks.DLinknet import DLinkNet34, DLinkNet50, DLinkNet101
from networks.Unet import Unet
from networks.Unet_new import UNet
from networks.Dunet import Dunet
from networks.Deeplab_v3_plus import DeepLabv3_plus
from networks.FCN8S import FCN8S
from networks.DABNet import DABNet
from networks.Segformer import Segformer
from networks.RS_Segformer import RS_Segformer
from networks.DE_Segformer import DE_Segformer

class SolverFrame():
    def __init__(self, net):
        self.net = net.cuda() # 模型迁移至显卡
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count())) # 支持多卡并行处理

    def load(self, path):
        self.net.load_state_dict(torch.load(path)) # 模型读取

class Predict():
    def __init__(self, net, class_number, band_num):
        self.class_number = class_number # 类别数
        self.img_mean = data_dict['mean'] # 数据集均值
        self.std = data_dict['std'] # 数据集方差
        self.net = net # 模型
        self.band_num = band_num # 影像波段数
    
    def Predict_wHy(self, img_block, dst_ds, xoff, yoff):
        img_block = img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32) # 数据类型转换

        self.net.eval() # 启动预测模式

        for i in range(self.band_num): # 数据标准化
            img_block[:, :, i] -= self.img_mean[i] 
        img_block = img_block / self.std

        img_block = np.expand_dims(img_block, 0) # 扩展数据维度 (h, w, c) -> (b, h, w, c) 
        img_block = img_block.transpose(0, 3, 1, 2) # (b, h, w, c) -> (b, c, h, w)
        img_block = Variable(torch.Tensor(img_block).cuda()) # Variable容器装载
        predict_out = self.net.forward(img_block).squeeze().cpu().data.numpy() # 模型预测；删除b维度；转换为numpy

        predict_out = predict_out.transpose(1, 2, 0) # (c, h, w) -> (h, w, c)
        predict_result = np.argmax(predict_out, axis=2) # 返回第三维度最大值的下标
        dst_ds.GetRasterBand(1).WriteArray(predict_result, xoff, yoff) # 预测结果写入gdal_dataset


    def Main(self, allpath, outpath, target_size=256, unify_read_img = False):  
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            dataset = gdal.Open(one_path) # GDAL打开待预测影像
            if dataset == None:
                print("failed to open img")
                sys.exit(1)
            img_width = dataset.RasterXSize # 读取影像宽度
            img_height = dataset.RasterYSize # 读取影像高度

            '''新建输出tif'''
            d, n = os.path.split(one_path)

            projinfo = dataset.GetProjection() # 获取原始影像投影
            geotransform = dataset.GetGeoTransform() # 获取原始影像地理坐标

            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 数据格式
            name = n[:-4] + '_result' + '.tif'  # 输出文件名

            dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                                1, gdal.GDT_Byte)  # 创建预测结果写入文件
            dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
            dst_ds.SetProjection(projinfo)  # 写入投影

            
            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray() # 影像一次性读入内存
                # 全局整体
                for i in tqdm(range(0, img_width-target_size, target_size)):
                    for j in range(0, img_height-target_size, target_size):
                        self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j)
                
                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)

                # 右下角
                self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size)
                
                dst_ds.FlushCache() # 全部预测完毕后统一写入磁盘
            else:
                '''分块读取影像并预测'''
                # 全局整体
                for i in tqdm(range(0, img_width-target_size, target_size)):
                    img_block = dataset.ReadAsArray(i, 0, target_size, dataset.RasterYSize) # 读取一列影像进内存
                    for j in range(0, img_height-target_size, target_size):
                        self.Predict_wHy(img_block[:, j:j+target_size, :].copy(), dst_ds, xoff=0, yoff=j)
                    dst_ds.FlushCache() # 预测完每列后写入磁盘
                    
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

    predictImgPath = r'E:\project_populus_UAV\1-clip_img\1-pretrain_img_ds025_240307' # 待预测影像的文件夹路径
    Img_type = '*.tif' # 待预测影像的类型
    trainListRoot = r'E:\project_populus_UAV\2-train_list\trainlist_pretrain_ds025_240307_1.txt' #与模型训练相同的训练列表路径
    num_class = 2 # 样本类别数
    model = UNet #模型
    model_path = r'E:\project_populus_UAV\3-weights\UNet_populus_ds025_0307_1.th' # 模型文件完整路径
    output_path = r'E:\project_populus_UAV\4-predict_result' # 输出的预测结果路径
    band_num = 4 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    target_size = 256 # 预测滑窗大小，应与训练集应一致
    unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True

    '''收集训练集信息'''
    dataCollect = DataTrainInform(classes_num=num_class, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
    data_dict = dataCollect.collectDataAndSave()
    # '''手动设置data_dict'''
    # data_dict = {}
    # data_dict['mean'] = [117.280266, 128.70387, 136.86803]
    # data_dict['std'] = [43.33161, 39.06087, 34.673794]
    # data_dict['classWeights'] = np.array([2.5911248, 3.8909917, 9.9005165, 9.21661, 7.058571, 10.126685, 3.4428556, 10.29797, 5.424672, 8.990792], dtype=np.float32)
    # data_dict['img_shape'] = [1024, 1024, 3]

    print('data mean: ', data_dict['mean'])
    print('data std: ', data_dict['std'])

    '''初始化模型'''
    solver = SolverFrame(net = model(num_classes=num_class, band_num=band_num))
    solver.load(model_path) # 加载模型
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    '''读取待预测影像'''
    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type) # 过滤对应文件类型
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    if not listpic:
        print('listpic is none')
        exit(1)
    else:
        print(listpic)

    '''执行预测'''
    predict_instantiation = Predict(net=solver.net, class_number=num_class, band_num=band_num) # 初始化预测
    predict_instantiation.Main(listpic, output_path, target_size, unify_read_img=unify_read_img) # 预测主体