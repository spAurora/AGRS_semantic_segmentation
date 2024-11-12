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

# from networks.DLinknet import DLinkNet34, DLinkNet50, DLinkNet101
# from networks.UNet_Light import UNet_Light
from networks.UNet import UNet
# from networks.DUNet import DUNet
# from networks.Deeplab_v3_plus import DeepLabv3_plus
# from networks.FCN8S import FCN8S
# from networks.DABNet import DABNet
# from networks.Segformer import Segformer
# from networks.RS_Segformer import RS_Segformer
# from networks.DE_Segformer import DE_Segformer
# from networks.HRNet import HRNet
# from networks.UNetFormer import UNetFormer
# from networks.HRNet import HRNet
# from networks.FCN import FCN_ResNet50, FCN_ResNet101
# from networks.U_MobileNet import U_MobileNet
# from networks.SegNet import SegNet
# from networks.U_ConvNeXt import U_ConvNeXt
# from networks.U_ConvNeXt_HWD import U_ConvNeXt_HWD
# from networks.U_ConvNeXt_HWD_DS import U_ConvNeXt_HWD_DS

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

    def Main(self, dataset, dst_ds, target_size=256, unify_read_img = False, overlap_rate = 0, if_mask=False, mask_path=''):  
        print('start predict...')

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
                    sys.exit(1)

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


if __name__ == '__main__':

    type_list = ['farmland', 'water', 'glacier','desert', 'building', 'forest_and_grass'] # 序列类别，按顺序预测
    target_size_list = [256, 512, 256, 256, 256, 256] # 预测尺寸集合，与类别对应
    predictImgPath = r'D:\DL_system_test\Show_area\1-test_img' # 待预测影像的文件夹路径
    Img_type = '*.tif' # 待预测影像的类型
    trainListRoot = r'D:\DL_system_test\Show_area\2-train_lists' #与模型训练相同的训练列表文件夹
    num_class = 2 # 样本类别数
    model = UNet #模型
    model_path = r'D:\DL_system_test\Show_area\3-models' # 模型文件夹路径
    output_path = r'D:\DL_system_test\Show_area\4-predict_result' # 输出的预测结果路径
    band_num = 4 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True
    overlap_rate = 0.1 # 滑窗间的重叠率

    if_mask = False # 是否开启mask模式；mask模式仅在unify_read_img==True时有效
    mask_path = r'E:\hami\mask' # mask路径 路径下需要有*.npz掩膜（./tools/generate_mask_by_moasic_line.py生成）

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    '''读取待预测影像集合'''
    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type) # 过滤对应文件类型
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    if not listpic:
        print('影像列表为空')
        exit(1)
    else:
        print(listpic)
    
    '''逐个读取影像'''
    for one_path in listpic:
        t0 = time.time()
        dataset = gdal.Open(one_path) # GDAL打开待预测影像
        if dataset == None:
            print("打开影像失败")
            sys.exit(1)
        img_width = dataset.RasterXSize # 读取影像宽度
        img_height = dataset.RasterYSize # 读取影像高度

        '''新建临时预测结果'''
        d, n = os.path.split(one_path)

        proj = dataset.GetProjection() # 获取原始影像投影
        geotransform = dataset.GetGeoTransform() # 获取原始影像地理坐标

        format = "GTiff"
        driver = gdal.GetDriverByName(format)  # 数据格式
        name = n[:-4] + '.tif'
        # name = n[:-4] + '_result' + '.tif'  # 输出文件名

        dst_ds = driver.Create(output_path + '/' + name, dataset.RasterXSize, dataset.RasterYSize,
                        1, gdal.GDT_Byte)  # 创建预测结果写入文件
        dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
        dst_ds.SetProjection(proj)  # 写入投影

        '''新建掩膜，新建最终预测结果'''
        image_height = dataset.RasterYSize
        image_width = dataset.RasterXSize
        mask = np.ones((image_height, image_width), dtype=np.uint8)
        final_predict_result = np.zeros((image_height, image_width), dtype=np.uint8)

        format = "GTiff"
        driver = gdal.GetDriverByName(format)  # 数据格式
        dst_ds_final = driver.Create(output_path + '/' + name[:-4] + '_final.tif', image_width, image_height,
                            1, gdal.GDT_Byte)  # 创建预测结果写入文件
        dst_ds_final.SetGeoTransform(geotransform)  # 写入地理坐标
        dst_ds_final.SetProjection(proj)  # 写入投影

        '''逐个影像处理'''
        for i in range(len(type_list)):
             
            '''更新参数'''
            model_full_path = model_path + '/' + type_list[i] + '.th' # 模型文件完整路径
            trainListRoot_full_path = trainListRoot + '/' + type_list[i] + '.txt' # 训练列表文件完整路径
            target_size = target_size_list[i] # 预测目标尺寸
            num_class = 2
            label_norm = True

            '''初始化模型'''
            if i == 5:
                num_class = 3
                label_norm = False
            solver = SolverFrame(net = model(num_classes=num_class, band_num=band_num))
            solver.load(model_full_path) # 加载模型

            # '''收集训练集信息'''
            # dataCollect = DataTrainInform(classes_num=num_class, trainlistPath=trainListRoot_full_path, band_num=band_num, label_norm=label_norm) #计算数据集信息
            # data_dict = dataCollect.collectDataAndSave()
            # print(data_dict) # 目视收集训练集信息
            
            '''手动设置data_dict'''
            data_dict_list = []
            # 耕地
            data_dict = {}
            data_dict['mean'] = [51.17147 , 52.451912, 54.48915, 88.72365]
            data_dict['std'] = [5.7302866, 6.9535646, 10.717318, 13.127798]
            data_dict['classWeights'] = np.array([1.5897559, 3.5610034], dtype=np.float32)
            data_dict['img_shape'] = [256, 256, 4]
            data_dict_list.append(data_dict)

            # 水体
            data_dict = {}
            data_dict['mean'] = [54.104237, 56.034992, 59.819637, 70.04928]
            data_dict['std'] = [16.814262, 18.240042 , 21.6526 , 25.394392]
            data_dict['classWeights'] = np.array([1.476799, 4.797466], dtype=np.float32)
            data_dict['img_shape'] = [512, 512, 4]
            data_dict_list.append(data_dict)

            # 冰川
            data_dict = {}
            data_dict['mean'] = [61.214104 , 62.2274 , 64.94447, 78.193565]
            data_dict['std'] = [21.497837, 22.392473 , 25.12916, 27.829441]
            data_dict['classWeights'] = np.array([1.4644064, 5.019629], dtype=np.float32)
            data_dict['img_shape'] = [256, 256, 4]
            data_dict_list.append(data_dict)                            

            # 沙漠
            data_dict = {}
            data_dict['mean'] = [56.313194, 59.86854 , 68.21926 , 82.18617]
            data_dict['std'] = [4.37653, 5.7297754 , 8.603879, 9.522306]
            data_dict['classWeights'] = np.array([1.9602377, 2.3354318], dtype=np.float32)
            data_dict['img_shape'] = [256, 256, 4]
            data_dict_list.append(data_dict)

            # 建筑 
            data_dict = {}
            data_dict['mean'] = [36.724808, 36.5781, 37.73344, 64.02281]
            data_dict['std'] = [11.520095, 12.19692, 15.445036, 20.324085]
            data_dict['classWeights'] = np.array([1.6566823, 3.1671135], dtype=np.float32)
            data_dict['img_shape'] = [256, 256, 4]
            data_dict_list.append(data_dict)

            # 森林和草地 
            data_dict = {}
            data_dict['mean'] = [45.332203, 45.01826, 44.3322, 75.280464]
            data_dict['std'] = [8.778328, 9.727278, 12.850872, 17.93026]
            data_dict['classWeights'] = np.array([1.8135905, 5.6533585, 3.170361], dtype=np.float32) # 注意这里是3个类别权重
            data_dict['img_shape'] = [256, 256, 4]
            data_dict_list.append(data_dict)

            data_dict = data_dict_list[i] # 读取data_dict

            '''执行预测'''
            predict_instantiation = Predict(net=solver.net, class_number=num_class, band_num=band_num) # 初始化预测
            predict_instantiation.Main(dataset=dataset, dst_ds=dst_ds, target_size=target_size, unify_read_img=unify_read_img, overlap_rate=overlap_rate, if_mask=if_mask, mask_path=mask_path) # 预测主体

            print('预测耗费时间: %0.1f(s).' % (time.time() - t0))

            '''执行完预测后更新掩膜和最终预测结果'''
            dataset_predicct_temp = gdal.Open(output_path + '/' + name) # GDAL打开当前预测结果
            if dataset_predicct_temp == None:
                print("读取预测结果失败")
                sys.exit(1)
            predict_temp = dataset_predicct_temp.ReadAsArray()

            if i != 5:
                # mask处理
                predict_temp *= mask
                # 更新mask
                mask = (~predict_temp) & mask
                # 更新最终预测结果
                final_predict_result += predict_temp*(i+1)
            else: # 针对森林和草地在一起的情况
                # mask处理
                predict_temp *= mask

                '''处理森林 假设森林值为1'''
                predict_temp_forest = predict_temp.copy()
                predict_temp_forest[predict_temp != 1] = 0

                # 更新mask
                mask = (~predict_temp_forest) & mask
                # 更新最终预测结果
                final_predict_result += predict_temp_forest*(i+1)

                '''处理草地 假设草地值为2'''
                predict_temp_grass = predict_temp.copy()
                predict_temp_grass[predict_temp != 2] = 0
                predict_temp_grass[predict_temp_grass == 2] = 1

                # 更新mask
                mask = (~predict_temp_grass) & mask
                # 更新最终预测结果
                final_predict_result += predict_temp_grass*(i+2)


        dst_ds_final.GetRasterBand(1).WriteArray(final_predict_result, 0, 0) #  写入预测结果
        dst_ds_final.FlushCache() # 全部预测完毕后统一刷新磁盘缓存
