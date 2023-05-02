# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型可视化
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import numpy as np
import os
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import fnmatch
import sys
import math

import matplotlib.pyplot as plt

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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

global_cnt = 0 # 全局变量global_cnt用于控制预测图像编码

def PrintEachFeatureMap(feature_map_out, enabled=False):
    save_path = r'C:\Users\75198\OneDrive\论文\SCI-3-3 Remote sensing data augmentation\图片\4-隐藏层可视化\LV2'
    datasets_name = r'ATSC+supermix_LV2'
    deep_level = r'x3'
    haze_level = r'lv2'

    feature_map_out = feature_map_out.squeeze() # 删除b维度
    feature_map_out = feature_map_out.cpu().detach().numpy() # tensor转numpy
    feature_map_number = np.shape(feature_map_out)[0] # 获取特征图数量

    fmap_list_uint8 = []
    for i in range(feature_map_number):
        max_value = np.max(feature_map_out[i])
        min_value = np.min(feature_map_out[i])
        if max_value - min_value == 0:
            fmap_list_uint8.append(np.zeros_like(feature_map_out[i]).astype(np.uint8))
        else:
            fmap_list_uint8.append((((feature_map_out[i] - min_value) / (max_value - min_value)) * 255).astype(np.uint8))
    
    print(len(fmap_list_uint8), feature_map_number)
    
    # 计算特征图排列方式
    k = int(math.ceil(math.sqrt(feature_map_number)))
    rows = math.ceil(feature_map_number / k)
    cols = k if feature_map_number >= (k-1) * rows else feature_map_number - k * (rows - 1)

    # 创建子图画布
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
    axs = axs.ravel() # 将二维数组展开成一维数组

    # 显示特征图
    for i in range(feature_map_number):
        axs[i].imshow(fmap_list_uint8[i], cmap='inferno')
        axs[i].axis('off') # 隐藏坐标轴
    
    global global_cnt
    global_cnt += 1

    plt.savefig(save_path + '/' + datasets_name+'_'+deep_level+'_'+haze_level+'-'+str(global_cnt)+'_fmap.png', dpi=300, bbox_inches='tight')

def modify_feats(feats):  # feaats形状: [b,c,h,w]

    return feats_mean

class SolverFrame():
    def __init__(self, net):
        self.net = net.cuda()  # 模型迁移至显卡
        self.net = torch.nn.DataParallel(
            self.net, device_ids=range(torch.cuda.device_count()))  # 支持多卡并行处理

    def load(self, path):
        self.net.load_state_dict(torch.load(path))  # 模型读取


class Predict():
    def __init__(self, net, class_number, band_num, output_shape):
        self.class_number = class_number  # 类别数
        self.img_mean = data_dict['mean']  # 数据集均值
        self.std = data_dict['std']  # 数据集方差
        self.net = net  # 模型
        self.band_num = band_num  # 影像波段数
        self.output_shape = output_shape

    def Predict_wHy(self, img_block, dst_ds, xoff, yoff):
        img_block = img_block.transpose(1, 2, 0)  # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32)  # 数据类型转换

        self.net.eval()  # 启动预测模式

        for i in range(self.band_num):  # 数据标准化
            img_block[:, :, i] -= self.img_mean[i]
        img_block = img_block / self.std

        # 扩展数据维度 (h, w, c) -> (b, h, w, c)
        img_block = np.expand_dims(img_block, 0)
        # (b, h, w, c) -> (b, c, h, w)
        img_block = img_block.transpose(0, 3, 1, 2)
        img_block = Variable(torch.Tensor(img_block).cuda())  # Variable容器装载
        
        _, feature_map_out = self.net.forward(img_block)  # 可视化数据输出
        vis_out = torch.mean(feature_map_out, dim=1, keepdim=True) # 特征图c轴均值
        vis_out = F.interpolate(vis_out, size=self.output_shape, mode='bilinear', align_corners=False) # 上采样到预测滑窗大小
        vis_out = vis_out.squeeze() # 删除1维度 (b, c, h, w) -> (h, w)
        vis_out = vis_out.cpu().detach().numpy() # tensor转numpy
        vis_out = (((vis_out - np.min(vis_out))/(np.max(vis_out)-np.min(vis_out)))*255).astype(np.uint8) # 拉伸到0-255之间
        
        dst_ds.GetRasterBand(1).WriteArray(vis_out, xoff, yoff) # 特征图结果写入gdal_dataset

        PrintEachFeatureMap(feature_map_out, enabled=True)

    def Main(self, allpath, outpath, target_size=256, unify_read_img=False):
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            dataset = gdal.Open(one_path)  # GDAL打开待预测影像
            if dataset == None:
                print("failed to open img")
                sys.exit(1)
            img_width = dataset.RasterXSize  # 读取影像宽度
            img_height = dataset.RasterYSize  # 读取影像高度

            '''新建输出tif'''
            d, n = os.path.split(one_path)

            projinfo = dataset.GetProjection()  # 获取原始影像投影
            geotransform = dataset.GetGeoTransform()  # 获取原始影像地理坐标

            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 数据格式
            name = n[:-4] + '_result' + '.tif'  # 输出文件名

            dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                                   1, gdal.GDT_Byte)  # 创建预测结果写入文件
            dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
            dst_ds.SetProjection(projinfo)  # 写入投影

            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray()  # 影像一次性读入内存
                # 全局整体
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    for j in range(0, img_height - target_size, target_size):
                        self.Predict_wHy(
                            img_block[:, j:j + target_size, i:i + target_size].copy(), dst_ds, xoff=i, yoff=j)

                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    self.Predict_wHy(img_block[:, row_begin:row_begin + target_size,
                                               i:i + target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    self.Predict_wHy(
                        img_block[:, j:j + target_size, col_begin:col_begin + target_size].copy(), dst_ds, xoff=col_begin, yoff=j)

                # 右下角
                self.Predict_wHy(img_block[:, row_begin:row_begin + target_size, col_begin:col_begin +
                                           target_size].copy(), dst_ds, img_width - target_size, img_height - target_size)

                dst_ds.FlushCache()  # 全部预测完毕后统一写入磁盘
            else:
                '''分块读取影像并预测'''
                # 全局整体
                for i in tqdm(range(0, math.floor(img_width / target_size - 1) * target_size, target_size)):
                    for j in range(0, math.floor(img_height / target_size - 1) * target_size, target_size):
                        img_block = dataset.ReadAsArray(
                            i, j, target_size, target_size)  # 读取滑窗影像进内存
                        self.Predict_wHy(img_block.copy(),
                                         dst_ds, xoff=i, yoff=j)
                    dst_ds.FlushCache()  # 预测完每列后写入磁盘

                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    img_block = dataset.ReadAsArray(
                        i, row_begin, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds,
                                     xoff=i, yoff=row_begin)
                dst_ds.FlushCache()

                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    img_block = dataset.ReadAsArray(
                        col_begin, j, target_size, target_size)
                    self.Predict_wHy(img_block.copy(), dst_ds,
                                     xoff=col_begin, yoff=j)
                dst_ds.FlushCache()

                # 右下角
                img_block = dataset.ReadAsArray(
                    img_width - target_size, img_height - target_size, target_size, target_size)
                self.Predict_wHy(img_block.copy(), dst_ds,
                                 img_width - target_size, img_height - target_size)
                dst_ds.FlushCache()

            print('预测耗费时间: %0.1f(s).' % (time.time() - t0))


if __name__ == '__main__':

    predictImgPath = r'C:\Users\75198\OneDrive\论文\SCI-3-3 Remote sensing data augmentation\图片\4-隐藏层可视化\0-clip_img'  # 待预测影像的文件夹路径
    Img_type = '*.tif'  # 待预测影像的类型
    trainListRoot = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-trainlist\3-trainlist_clear_mix_sim_haze_ATSC+supermix_LV2_rate_0.2_230425.txt'  # 与模型训练相同的训练列表路径
    numclass = 3  # 样本类别数
    model = UNet  # 模型
    model_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\3-weights\4-Unet-huyang_clear_mix_sim_haze_ATSC+supermix_LV2_rate_0.2_230428.th'  # 模型文件完整路径
    output_path = r'C:\Users\75198\OneDrive\论文\SCI-3-3 Remote sensing data augmentation\图片\4-隐藏层可视化\LV2'  # 输出的预测结果路径
    band_num = 8  # 影像的波段数 训练与预测应一致
    label_norm = False  # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    target_size = 256  # 预测滑窗大小，应与训练集应一致
    unify_read_img = True  # 是否集中读取影像并预测 内存充足的情况下尽量设置为True
    if_vis = True  # 是否输出中间可视化信息 可视化文件中必须设置为True

    '''收集训练集信息'''
    dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot,
                                  band_num=band_num, label_norm=label_norm)  # 计算数据集信息
    data_dict = dataCollect.collectDataAndSave()

    print('data mean: ', data_dict['mean'])
    print('data std: ', data_dict['std'])

    '''初始化模型'''
    solver = SolverFrame(net=model(num_classes=numclass,
                                   band_num=band_num, ifVis=if_vis))
    solver.load(model_path)  # 加载模型
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    '''读取待预测影像'''
    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type)  # 过滤对应文件类型
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])

    if not listpic:
        print('listpic is none')
        exit(1)
    else:
        print(listpic)

    '''执行预测'''
    predict_instantiation = Predict(
        net=solver.net, class_number=numclass, band_num=band_num, output_shape=(target_size, target_size))  # 初始化预测
    predict_instantiation.Main(
        listpic, output_path, target_size, unify_read_img=unify_read_img)  # 预测主体
