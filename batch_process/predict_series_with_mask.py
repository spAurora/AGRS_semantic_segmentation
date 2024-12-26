# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型序列掩膜预测
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
from networks.UNet_Light import UNet_Light
from networks.UNet import UNet
from networks.DUNet import DUNet
from networks.Deeplab_v3_plus import DeepLabv3_plus
from networks.FCN8S import FCN8S
from networks.DABNet import DABNet
from networks.Segformer import Segformer
from networks.RS_Segformer import RS_Segformer
from networks.DE_Segformer import DE_Segformer
from networks.HRNet import HRNet
from networks.UNetFormer import UNetFormer
from networks.HRNet import HRNet
from networks.FCN import FCN_ResNet50, FCN_ResNet101
from networks.U_MobileNet import U_MobileNet
from networks.SegNet import SegNet
from networks.U_ConvNeXt import U_ConvNeXt
from networks.U_ConvNeXt_HWD import U_ConvNeXt_HWD
from networks.U_ConvNeXt_HWD_DS import U_ConvNeXt_HWD_DS

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
    
    def Predict_wHy(self, img_block, dst_ds, xoff, yoff, overlap_rate = 0):
        img_block = img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        img_block = img_block.astype(np.float32) # 数据类型转换
        block_width = np.size(img_block, 0)

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
        
        # dst_ds.GetRasterBand(1).WriteArray(predict_result[int(block_width*overlap_rate):int(block_width*(1-overlap_rate)), int(block_width*overlap_rate):int(block_width*(1-overlap_rate))], xoff+int(block_width*overlap_rate), yoff+int(block_width*overlap_rate)) # 预测结果写入gdal_dataset

        return predict_result

    def Main(self, allpath, outpath, target_size=256, unify_read_img = False, overlap_rate = 0, if_mask=False, mask_path=''):  
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
            name = n[:-4] + '.tif'
            # name = n[:-4] + '_result' + '.tif'  # 输出文件名

            dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                                1, gdal.GDT_Byte)  # 创建预测结果写入文件
            dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
            dst_ds.SetProjection(projinfo)  # 写入投影

            step = int(target_size * (1-2*overlap_rate)) # overlap_rate控制步长

            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray() # 影像一次性读入内存

                predict_result_all = np.zeros((img_height, img_width), dtype=np.uint8)
                
                if if_mask: # 读取掩膜
                    mask_full_path = mask_path + '/' + n[:-4] + '.npz'
                    if os.path.exists(mask_full_path):
                        m_data = np.load(mask_path + '/' + n[:-4] + '.npz')
                        mask = m_data['mask']
                    else:
                        print('does not exist: ' + mask_full_path)
                        continue # 如果mask文件不存在直接跳过该文件

                # 上侧边缘
                row_begin = 0
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    if if_mask: # 掩膜模式
                        mask_patch = mask[row_begin:row_begin+target_size, i:i+target_size].copy()
                        if np.all(mask_patch == 0): # 如果掩膜为空，直接跳过
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)
                            predict_patch = predict_patch * mask_patch # 掩膜处理
                            predict_result_all[row_begin:row_begin+target_size, i:i+target_size] = predict_patch.copy()
                    else: # 非掩膜模式
                        predict_result_all[row_begin:row_begin+target_size, i:i+target_size] = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    if if_mask: # 掩膜模式
                        mask_patch = mask[row_begin:row_begin+target_size, i:i+target_size].copy()
                        if np.all(mask_patch == 0): # 如果掩膜为空，直接跳过
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)
                            predict_patch = predict_patch * mask_patch # 掩膜处理
                            predict_result_all[row_begin:row_begin+target_size, i:i+target_size] = predict_patch.copy()
                    else: # 非掩膜模式
                        predict_result_all[row_begin:row_begin+target_size, i:i+target_size] = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)

                # 左侧边缘
                col_begin = 0
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    if if_mask: # 掩膜模式
                        mask_patch = mask[j:j+target_size, col_begin:col_begin+target_size].copy()
                        if np.all(mask_patch == 0): # 如果掩膜为空，直接跳过
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)
                            predict_patch = predict_patch * mask_patch # 掩膜处理
                            predict_result_all[j:j+target_size, col_begin:col_begin+target_size] = predict_patch.copy()
                    else: # 非掩膜模式
                        predict_result_all[j:j+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)
                    
                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    if if_mask: # 掩膜模式
                        mask_patch = mask[j:j+target_size, col_begin:col_begin+target_size].copy()
                        if np.all(mask_patch == 0): # 如果掩膜为空，直接跳过
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)
                            predict_patch = predict_patch * mask_patch # 掩膜处理
                            predict_result_all[j:j+target_size, col_begin:col_begin+target_size] = predict_patch.copy()
                    else: # 非掩膜模式
                        predict_result_all[j:j+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(img_block[:, j:j+target_size, col_begin:col_begin+target_size].copy(), dst_ds, xoff=col_begin, yoff=j)

                # 右下角
                if if_mask: # 掩膜模式
                    mask_patch = mask[row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy()
                    predict_patch = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size)
                    predict_patch = predict_patch * mask_patch # 直接掩膜处理
                    predict_result_all[row_begin:row_begin+target_size, col_begin:col_begin+target_size] = predict_patch.copy()
                else:
                    predict_result_all[row_begin:row_begin+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(img_block[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size].copy(), dst_ds, img_width-target_size, img_height-target_size)
                
                # 全局整体
                for i in tqdm(range(0, img_width-target_size, step)):
                    for j in range(0, img_height-target_size, step):
                        if if_mask: # 掩膜模式
                            mask_patch = mask[j:j+target_size, i:i+target_size].copy()
                            if np.all(mask_patch == 0): # 如果掩膜为空，直接跳过
                                continue
                            else:
                                predict_patch = self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j, overlap_rate=overlap_rate)
                                predict_patch = predict_patch * mask_patch # 掩膜处理
                                predict_result_all[j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), i+int(target_size*overlap_rate):i+int(target_size*(1-overlap_rate))] = predict_patch[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))]
                        else: # 非掩膜模式
                            predict_patch = self.Predict_wHy(img_block[:, j:j+target_size, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j, overlap_rate=overlap_rate)
                            predict_result_all[j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), i+int(target_size*overlap_rate):i+int(target_size*(1-overlap_rate))] = predict_patch[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))]
                    
                dst_ds.GetRasterBand(1).WriteArray(predict_result_all, 0, 0)
                dst_ds.FlushCache() # 全部预测完毕后统一刷新磁盘缓存

            else:
                '''分块读取影像并预测'''
                '''设备内存过小或者影像过大时应用该模式'''
                predict_result_col = np.zeros((img_height, target_size), dtype=np.uint8)
                predict_result_row = np.zeros((target_size, img_width), dtype=np.uint8)                
                # 上侧边缘
                row_begin = 0
                img_block = dataset.ReadAsArray(0, row_begin, dataset.RasterXSize, target_size)
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    predict_result_row[:, i:i+target_size] = self.Predict_wHy(img_block[:, :, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_row, 0, row_begin)
                dst_ds.FlushCache()
                    
                # 下侧边缘
                row_begin = img_height - target_size
                img_block = dataset.ReadAsArray(0, row_begin, dataset.RasterXSize, target_size)
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    predict_result_row[:, i:i+target_size] = self.Predict_wHy(img_block[:, :, i:i+target_size].copy(), dst_ds, xoff=i, yoff=row_begin)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_row, 0, row_begin)
                dst_ds.FlushCache()

                # 左侧边缘
                col_begin = 0
                img_block = dataset.ReadAsArray(col_begin, 0, target_size, dataset.RasterYSize)
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    predict_result_col[j:j+target_size, :] = self.Predict_wHy(img_block[:,j:j+target_size,:].copy(), dst_ds, xoff=col_begin, yoff=j)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_col, col_begin, 0)
                dst_ds.FlushCache()

                # 右侧边缘
                col_begin = 0
                img_block = dataset.ReadAsArray(col_begin, 0, target_size, dataset.RasterYSize)
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    predict_result_col[j:j+target_size, :] = self.Predict_wHy(img_block[:,j:j+target_size,:].copy(), dst_ds, xoff=col_begin, yoff=j)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_col, col_begin, 0)
                dst_ds.FlushCache()

                # 右下角
                img_block = dataset.ReadAsArray(img_width-target_size, img_height-target_size, target_size, target_size)
                predict_result = self.Predict_wHy(img_block.copy(), dst_ds, img_width-target_size, img_height-target_size)
                dst_ds.GetRasterBand(1).WriteArray(predict_result, img_width-target_size, img_height-target_size)
                dst_ds.FlushCache()

                # 全局整体
                for i in tqdm(range(0, img_width-target_size, step)):
                    img_block = dataset.ReadAsArray(i, 0, target_size, dataset.RasterYSize) # 读取一列影像进内存
                    for j in range(0, img_height-target_size, step):
                        predict_result = self.Predict_wHy(img_block[:, j:j+target_size, :].copy(), dst_ds, xoff=i, yoff=j, overlap_rate=overlap_rate)
                        predict_result_col[j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))] = predict_result[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))]
                    dst_ds.GetRasterBand(1).WriteArray(predict_result_col[int(target_size*overlap_rate):img_height-int(target_size*overlap_rate) ,int(target_size*overlap_rate):int(target_size*(1-overlap_rate))], i+int(target_size*overlap_rate), int(target_size*overlap_rate))
                dst_ds.FlushCache() # 最后刷新磁盘缓存

            print('预测耗费时间: %0.1f(s).' % (time.time() - t0))

if __name__ == '__main__':

    trainListRoot_list = []
    model_list = []
    model_name_list = []

    image_path = 'xxx.tif'
    final_predict_result_path = 'xxx.tif'

    # 读取待预测影像
    dataset_oriimage = gdal.Open(image_path) # GDAL打开当前预测结果
    if dataset_oriimage == None:
        print("failed to open ori image")
        sys.exit(1)
    proj = dataset_oriimage.GetProjection() # 获取原始影像投影
    geotransform = dataset_oriimage.GetGeoTransform() # 获取原始影像地理坐标

    # 新建掩膜，新建最终预测结果
    image_height = dataset_oriimage.RasterYSize
    image_width = dataset_oriimage.RasterXSize
    mask = np.ones((image_height, image_width), dtype=np.uint8)
    final_predict_result = np.zeros((image_height, image_width), dtype=np.uint8)

    format = "GTiff"
    driver = gdal.GetDriverByName(format)  # 数据格式
    dst_ds = driver.Create(final_predict_result_path, image_width, image_height,
                        1, gdal.GDT_Byte)  # 创建预测结果写入文件
    dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
    dst_ds.SetProjection(proj)  # 写入投影

    '''循环序列预测'''
    for i in range(5):
        predictImgPath = r'E:\project_hami_limuceng\0-srimg\8bit' # 待预测影像的文件夹路径
        Img_type = '*.tif' # 待预测影像的类型
        trainListRoot = r'E:\project_hami_limuceng\2-trainlist\train_list_240617.txt' #与模型训练相同的训练列表路径
        num_class = 2 # 样本类别数
        model = U_ConvNeXt_HWD_DS #模型
        model_path = r'E:\project_hami_limuceng\3-weights\U_ConvNeXt_HWD_DS_240617.th' # 模型文件完整路径
        output_path = r'E:\project_hami_limuceng\4-predict_result' # 输出的预测结果路径
        band_num = 4 #影像的波段数 训练与预测应一致
        label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
        target_size = 192 # 预测滑窗大小，应与训练集应一致
        unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True
        overlap_rate = 0.1 # 滑窗间的重叠率

        if_mask = False # 是否开启mask模式；mask模式仅在unify_read_img==True时有效
        mask_path = r'E:\hami\mask' # mask路径 路径下需要有*.npz掩膜（./tools/generate_mask_by_moasic_line.py生成）

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
        predict_instantiation.Main(listpic, output_path, target_size, unify_read_img=unify_read_img, overlap_rate=overlap_rate, if_mask=if_mask, mask_path=mask_path) # 预测主体

        # 执行完预测后更新掩膜和最终预测结果
        dataset = gdal.Open(listpic[0]) # GDAL打开当前预测结果
        if dataset == None:
            print("failed to open predict result")
            sys.exit(1)
        predict_temp = dataset.ReadAsArray()

        # mask处理
        predict_temp *= mask
        # 更新mask
        mask = (~predict_temp) and mask
        # 更新最终预测结果
        final_predict_result += predict_temp*(i+1)

    dst_ds.GetRasterBand(1).WriteArray(final_predict_result, 0, 0)
    dst_ds.FlushCache() # 全部预测完毕后统一刷新磁盘缓存
