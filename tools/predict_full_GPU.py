# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型预测
影像会整体迁移到显卡上进行推理
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
import mmap

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
# from networks.U_ConvNeXt_HWD import U_ConvNeXt_HWD
# from networks.U_ConvNeXt_HWD_DS import U_ConvNeXt_HWD_DS

from networks.MAE_Seg import MAEViTSegmentation
import cProfile

from numba import njit

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
        self.mean_tensor = torch.tensor(data_dict['mean'], dtype=torch.float32, device='cuda').view(self.band_num, 1, 1)
        self.std_tensor = torch.tensor(data_dict['std'], dtype=torch.float32, device='cuda').view(self.band_num, 1, 1)

        self.net.eval() # 启动预测模式
        torch.set_grad_enabled(False)
    
    def Predict_wHy(self, img_block):
        # 转换为GPU Tensor并在传输上启用异步（核心优化）
        img_tensor = img_block
        
        # GPU上执行向量化标准化（避免CPU循环）
        img_tensor = (img_tensor - self.mean_tensor) / self.std_tensor
        
        # 添加批次维度并保持通道优先
        img_tensor = img_tensor.unsqueeze(0)  # (1, c, h, w)
        
        # 模型推理
        predict_out = self.net(img_tensor)  # (1, c, h, w)
        
        # GPU上执行argmax（避免数据传输）
        predict_result = torch.argmax(predict_out.squeeze(0), dim=0)  # (h, w)
        
        # 返回CPU上的numpy数组（保持原始函数输出格式）
        return predict_result

    def Main(self, allpath, outpath, target_size=256, unify_read_img = False, overlap_rate = 0, if_mask=False, mask_path='', img_data_type=gdal.GDT_Byte):  
        print('start predict...')

        if img_data_type == gdal.GDT_Byte:
            dtype = np.uint8
        elif img_data_type == gdal.GDT_UInt16:
            dtype = np.uint16
        else:
            raise ValueError("Unsupported data type")

        for one_path in allpath:

            if if_mask: # 读取掩膜
                mask_full_path = mask_path + '/' + os.path.basename(one_path)[:-4] + '.npz'
                if os.path.exists(mask_full_path):
                    m_data = np.load(mask_path + '/' + os.path.basename(one_path)[:-4] + '.npz')
                    mask = m_data['mask']
                else:
                    print('Mask file does not exist: ' + mask_full_path)
                    continue # 如果mask文件不存在直接跳过该文件

            t0 = time.time()

            if if_vismem:
                # 获取影像文件名前缀（不包括扩展名）
                filename_prefix = os.path.splitext(os.path.basename(one_path))[0]

                input_dir = os.path.dirname(one_path)
                # 搜索配套文件（除主影像文件外其他相同前缀的文件）
                auxiliary_files = [
                    f for f in os.listdir(input_dir)
                    if f.startswith(filename_prefix) and f != os.path.basename(one_path)
                ]

                # 将主影像文件映射到内存
                with open(one_path, "rb") as f:
                    mmapped_main_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    gdal.FileFromMemBuffer("/vsimem/" + os.path.basename(one_path), mmapped_main_file[:])

                # 将配套文件映射到内存
                for aux_file in auxiliary_files:
                    aux_path = os.path.join(input_dir, aux_file)
                    with open(aux_path, "rb") as f_aux:
                        mmapped_aux_file = mmap.mmap(f_aux.fileno(), 0, access=mmap.ACCESS_READ)
                        gdal.FileFromMemBuffer(f"/vsimem/{aux_file}", mmapped_aux_file[:])

                dataset = gdal.Open("/vsimem/" + os.path.basename(one_path), gdal.GA_ReadOnly) # GDAL打开待预测影像
            else:
                dataset = gdal.Open(one_path, gdal.GA_ReadOnly) # GDAL打开待预测影像
            
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

            dst_ds = driver.Create(
                os.path.join(outpath, name),
                dataset.RasterXSize,
                dataset.RasterYSize,
                1,
                img_data_type,
                options=[
                    "TILED=YES",  # 启用平铺
                    "BLOCKXSIZE=1024",  # 设置平铺的宽度
                    "BLOCKYSIZE=1024",  # 设置平铺的高度
                    "COMPRESS=DEFLATE",  # 设置使用DEFLATE压缩算法
                ],
            )  # 创建预测结果写入文件
            dst_ds.GetRasterBand(1).SetNoDataValue(0)  # 将背景值设置为NODATA
            dst_ds.SetGeoTransform(geotransform)  # 写入地理坐标
            dst_ds.SetProjection(projinfo)  # 写入投影

            step = int(target_size * (1-2*overlap_rate)) # overlap_rate控制步长

            if unify_read_img:
                '''集中读取影像并预测'''
                img_block = dataset.ReadAsArray() # 影像一次性读入内存

                torch.cuda.empty_cache()
                
                # 根据dtype自动选择GPU张量类型
                if dtype == np.uint8:
                    torch_dtype = torch.uint8
                elif dtype == np.uint16:
                    torch_dtype = torch.int16  # PyTorch没有torch.uint16，使用int16代替
                else:
                    torch_dtype = torch.uint8  # 默认类型

                # 将整个图像数据一次性传输到GPU（核心优化）
                img_block_gpu = torch.from_numpy(img_block).to('cuda')
                
                # 创建GPU上的结果张量
                predict_result_all_tensor = torch.zeros((img_height, img_width), dtype=torch_dtype, device='cuda')
                
                # 如果在掩膜模式，将掩膜也移到GPU
                if if_mask:
                    mask_gpu = torch.from_numpy(mask).to('cuda')
                
                predict_result_all = np.zeros((img_height, img_width), dtype=dtype)

                # 上侧边缘
                row_begin = 0
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    if if_mask: # 掩膜模式
                        mask_patch = mask_gpu[row_begin:row_begin+target_size, i:i+target_size]  # GPU直接切片
                        if torch.all(mask_patch == 0): # GPU上判断
                            continue
                        else:
                            # 直接从GPU内存切片，避免CPU拷贝
                            predict_patch = self.Predict_wHy(img_block_gpu[:, row_begin:row_begin+target_size, i:i+target_size])
                            predict_patch = predict_patch * mask_patch # GPU上掩膜处理
                            predict_result_all_tensor[row_begin:row_begin+target_size, i:i+target_size] = predict_patch
                    else: # 非掩膜模式
                        predict_result_all_tensor[row_begin:row_begin+target_size, i:i+target_size] = self.Predict_wHy(
                            img_block_gpu[:, row_begin:row_begin+target_size, i:i+target_size]
                        )

                # 下侧边缘
                row_begin = img_height - target_size
                for i in tqdm(range(0, img_width - target_size, target_size)):
                    if if_mask:
                        mask_patch = mask_gpu[row_begin:row_begin+target_size, i:i+target_size]
                        if torch.all(mask_patch == 0):
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block_gpu[:, row_begin:row_begin+target_size, i:i+target_size])
                            predict_patch = predict_patch * mask_patch
                            predict_result_all_tensor[row_begin:row_begin+target_size, i:i+target_size] = predict_patch
                    else:
                        predict_result_all_tensor[row_begin:row_begin+target_size, i:i+target_size] = self.Predict_wHy(
                            img_block_gpu[:, row_begin:row_begin+target_size, i:i+target_size]
                        )

                # 左侧边缘
                col_begin = 0
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    if if_mask:
                        mask_patch = mask_gpu[j:j+target_size, col_begin:col_begin+target_size]
                        if torch.all(mask_patch == 0):
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block_gpu[:, j:j+target_size, col_begin:col_begin+target_size])
                            predict_patch = predict_patch * mask_patch
                            predict_result_all_tensor[j:j+target_size, col_begin:col_begin+target_size] = predict_patch
                    else:
                        predict_result_all_tensor[j:j+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(
                            img_block_gpu[:, j:j+target_size, col_begin:col_begin+target_size]
                        )
                            
                # 右侧边缘
                col_begin = img_width - target_size
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    if if_mask:
                        mask_patch = mask_gpu[j:j+target_size, col_begin:col_begin+target_size]
                        if torch.all(mask_patch == 0):
                            continue
                        else:
                            predict_patch = self.Predict_wHy(img_block_gpu[:, j:j+target_size, col_begin:col_begin+target_size])
                            predict_patch = predict_patch * mask_patch
                            predict_result_all_tensor[j:j+target_size, col_begin:col_begin+target_size] = predict_patch
                    else:
                        predict_result_all_tensor[j:j+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(
                            img_block_gpu[:, j:j+target_size, col_begin:col_begin+target_size]
                        )

                # 右下角
                if if_mask:
                    mask_patch = mask_gpu[row_begin:row_begin+target_size, col_begin:col_begin+target_size]
                    predict_patch = self.Predict_wHy(img_block_gpu[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size])
                    predict_patch = predict_patch * mask_patch
                    predict_result_all_tensor[row_begin:row_begin+target_size, col_begin:col_begin+target_size] = predict_patch
                else:
                    predict_result_all_tensor[row_begin:row_begin+target_size, col_begin:col_begin+target_size] = self.Predict_wHy(
                        img_block_gpu[:, row_begin:row_begin+target_size, col_begin:col_begin+target_size]
                    )
                
                # 全局整体
                for i in tqdm(range(0, img_width-target_size, step)):
                    for j in range(0, img_height-target_size, step):
                        if if_mask:
                            mask_patch = mask_gpu[j:j+target_size, i:i+target_size]
                            if torch.all(mask_patch == 0):
                                continue
                            else:
                                predict_patch = self.Predict_wHy(img_block_gpu[:, j:j+target_size, i:i+target_size])
                                predict_patch = predict_patch * mask_patch
                                predict_result_all_tensor[
                                    j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), 
                                    i+int(target_size*overlap_rate):i+int(target_size*(1-overlap_rate))
                                ] = predict_patch[
                                    int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), 
                                    int(target_size*overlap_rate):int(target_size*(1-overlap_rate))
                                ]
                        else:
                            predict_result_all_tensor[
                                j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), 
                                i+int(target_size*overlap_rate):i+int(target_size*(1-overlap_rate))
                            ] = self.Predict_wHy(img_block_gpu[:, j:j+target_size, i:i+target_size]
                            )[
                                int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), 
                                int(target_size*overlap_rate):int(target_size*(1-overlap_rate))
                            ]
                
                # 一次性将GPU结果转回CPU
                predict_result_all = predict_result_all_tensor.cpu().numpy()
                
                del img_block_gpu, predict_result_all_tensor
                if if_mask:
                    del mask_gpu

                dst_ds.GetRasterBand(1).WriteArray(predict_result_all, 0, 0)
                dst_ds.FlushCache()

            else:
                '''分块读取影像并预测'''
                '''设备内存过小或者影像过大时应用该模式'''
                predict_result_col = np.zeros((img_height, target_size), dtype=dtype)
                predict_result_row = np.zeros((target_size, img_width), dtype=dtype)

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

                # 右下角
                img_block = dataset.ReadAsArray(img_width-target_size, img_height-target_size, target_size, target_size)
                predict_result = self.Predict_wHy(img_block.copy(), dst_ds, img_width-target_size, img_height-target_size)
                dst_ds.GetRasterBand(1).WriteArray(predict_result, img_width-target_size, img_height-target_size)
                dst_ds.FlushCache()

                # 全局整体(按行读取, 与按列读取冲突，影像为压缩tif带金字塔时使用)
                for j in tqdm(range(0, img_height-target_size, step)):
                    img_block = dataset.ReadAsArray(0, j, dataset.RasterXSize, target_size) # 读取一行影像进内存
                    for i in range(0, img_width-target_size, step):
                        predict_result = self.Predict_wHy(img_block[:, :, i:i+target_size].copy(), dst_ds, xoff=i, yoff=j, overlap_rate=overlap_rate)
                        predict_result_row[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), i+int(target_size*overlap_rate):i+int(target_size*(1-overlap_rate))] = predict_result[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))]
                    dst_ds.GetRasterBand(1).WriteArray(predict_result_row[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):img_width-int(target_size*overlap_rate)], int(target_size*overlap_rate), j+int(target_size*overlap_rate))
                dst_ds.FlushCache() # 最后刷新磁盘缓存

                # # 全局整体(按列读取，与按行读取冲突)
                # for i in tqdm(range(0, img_width-target_size, step)):
                #     img_block = dataset.ReadAsArray(i, 0, target_size, dataset.RasterYSize) # 读取一列影像进内存
                #     for j in range(0, img_height-target_size, step):
                #         predict_result = self.Predict_wHy(img_block[:, j:j+target_size, :].copy(), dst_ds, xoff=i, yoff=j, overlap_rate=overlap_rate)
                #         predict_result_col[j+int(target_size*overlap_rate):j+int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))] = predict_result[int(target_size*overlap_rate):int(target_size*(1-overlap_rate)), int(target_size*overlap_rate):int(target_size*(1-overlap_rate))]
                #     dst_ds.GetRasterBand(1).WriteArray(predict_result_col[int(target_size*overlap_rate):img_height-int(target_size*overlap_rate) ,int(target_size*overlap_rate):int(target_size*(1-overlap_rate))], i+int(target_size*overlap_rate), int(target_size*overlap_rate))
                # dst_ds.FlushCache() # 最后刷新磁盘缓存

                # 左侧边缘
                col_begin = 0
                img_block = dataset.ReadAsArray(col_begin, 0, target_size, dataset.RasterYSize)
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    predict_result_col[j:j+target_size, :] = self.Predict_wHy(img_block[:,j:j+target_size,:].copy(), dst_ds, xoff=col_begin, yoff=j)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_col, col_begin, 0)
                dst_ds.FlushCache()

                # 右侧边缘
                col_begin = img_width - target_size
                img_block = dataset.ReadAsArray(col_begin, 0, target_size, dataset.RasterYSize)
                for j in tqdm(range(0, img_height - target_size, target_size)):
                    predict_result_col[j:j+target_size, :] = self.Predict_wHy(img_block[:,j:j+target_size,:].copy(), dst_ds, xoff=col_begin, yoff=j)
                dst_ds.GetRasterBand(1).WriteArray(predict_result_col, col_begin, 0)
                dst_ds.FlushCache()

            # 清理内存文件
            if if_vismem:
                gdal.Unlink("/vsimem/" + os.path.basename(one_path))
                for aux_file in auxiliary_files:
                    gdal.Unlink(f"/vsimem/{aux_file}")

            print('预测耗时: %0.1f(s).' % (time.time() - t0))

if __name__ == '__main__':

    predictImgPath = r'G:\PROJECT_GLOBAL_POPULUS_DATA_03\FQ-XJ_Nanjiang6_Tarim\IMAGE-FUSE' # 待预测影像的文件夹路径
    Img_type = '*.img' # 待预测影像的类型
    trainListRoot = r'G:\PROJECT_GLOBAL_POPULUS_SS_03\2-train_list\populus_250210.txt' #与模型训练相同的训练列表路径
    num_class = 2 # 样本类别数
    model = UNet #模型
    model_path = r'G:\PROJECT_GLOBAL_POPULUS_SS_03\3-weights\UNet_populus_250210.th' # 模型文件完整路径
    output_path = r'G:\PROJECT_GLOBAL_POPULUS_RESULT_03\raster_result_250703' # 输出的预测结果路径
    band_num = 4 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致
    target_size = 256 # 预测滑窗大小，应与训练集应一致
    unify_read_img = True # 是否集中读取影像并预测 内存充足的情况下尽量设置为True
    overlap_rate = 0.1 # 滑窗间的重叠率

    img_data_type = gdal.GDT_Byte # only support gdal.GDT_Byte or gdal.GDT_UInt16

    if_mask = True # 是否开启mask模式；mask模式仅在unify_read_img==True时有效
    mask_path = r'G:\PROJECT_GLOBAL_POPULUS_DATA_03\FQ-XJ_Nanjiang6_Tarim\MASK' # mask路径 路径下需要有*.npz掩膜（./tools/generate_mask_by_moasic_line.py生成）

    if_vismem = True # 是否开启虚拟文件系统; 开启后可大幅提高机械硬盘中的影像读取速度，但需要保证内存充足

    # '''收集训练集信息'''
    # dataCollect = DataTrainInform(classes_num=num_class, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
    # data_dict = dataCollect.collectDataAndSave()
    '''手动设置data_dict'''
    data_dict = {}
    data_dict['mean'] = [[39.528656, 42.048725, 45.86413,  61.985355]]
    data_dict['std'] = [ 5.203125,   7.0083585,  9.489312,  10.036979 ]
    data_dict['classWeights'] = np.array([1,1], dtype=np.float32)
    data_dict['img_shape'] = [256, 256, 4]

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
    predict_instantiation.Main(listpic, output_path, target_size, unify_read_img=unify_read_img, overlap_rate=overlap_rate, if_mask=if_mask, mask_path=mask_path, img_data_type=img_data_type) # 预测主体
