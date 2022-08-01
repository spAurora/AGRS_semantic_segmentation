# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
模型预测
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import skimage.io
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
    def __init__(self, net, data_dict, band_num = 3):
        self.net = net.cuda()
        self.img_mean = data_dict['mean']
        self.std = data_dict['std']
        self.band_num = band_num


        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def predict_x(self, img):
        self.net.eval()

        for i in range(self.band_num):
            img[:,:,i] -= self.img_mean[i]
        img = img / self.std

        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)
        img = V(torch.Tensor(img).cuda())
        maska = self.net.forward(img).squeeze().cpu().data.numpy()

        return maska

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

class Predict():
    def __init__(self, class_number):
        self.class_number = class_number

    def CreatTf(self, file_path_img, data, outpath):  # 创建tif文件
        print(file_path_img)
        d, n = os.path.split(file_path_img)
        dataset = gdal.Open(file_path_img, GA_ReadOnly)  

        projinfo = dataset.GetProjection() 
        geotransform = dataset.GetGeoTransform()

        format = "GTiff"
        driver = gdal.GetDriverByName(format)  # 数据格式
        name = n[:-4] + '_result' + '.tif'  # 输出文件名字


        dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                               1, gdal.GDT_Byte)  # 创建一个新的文件
        dst_ds.SetGeoTransform(geotransform)  # 写入投影
        dst_ds.SetProjection(projinfo)  # 写入坐标
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.FlushCache()
    
    def make_prediction_wHy(self, x, target_size, overlap_rate, predict, class_num):
        weights = np.zeros((x.shape[0], x.shape[1], class_num), dtype=np.float32)
        space = int(target_size * (1-overlap_rate))
        print('space: ', space)
        print('img shape: ', x.shape[0], x.shape[1], x.shape[2])
        pad_y = np.zeros(
            (x.shape[0], x.shape[1], class_num),
            dtype=np.float32)
        print('pad_y shape:', np.shape(pad_y))
        
        for i in tqdm(range(0, x.shape[0] - target_size, space)):
            for j in range(0, x.shape[1] - target_size, space):
                img_one = x[i:i + target_size, j:j + target_size, :]
                pre_one = predict(img_one)
                pre_one = pre_one.transpose(1,2,0)
                weight = weights[i:i + target_size, j:j + target_size]
                pre_current = pad_y[i:i + target_size, j:j + target_size]
                result = (weight * pre_current + pre_one) * (1 / (weight + 1))
                pad_y[i:i + target_size, j:j + target_size] = result
                weights[i:i + target_size, j:j + target_size] += 1

        col_begin = x.shape[1] - target_size
        for i in tqdm(range(0, x.shape[0] - target_size, target_size)):
            img_one = x[i:i + target_size, col_begin:x.shape[1], :]
            pre_one = predict(img_one)
            pre_one = pre_one.transpose(1, 2, 0)
            weight = weights[i:i + target_size, col_begin:x.shape[1]]
            pre_current = pad_y[i:i + target_size, col_begin:x.shape[1]]
            result = (weight * pre_current + pre_one) * (1 / (weight + 1))
            pad_y[i:i + target_size, col_begin:x.shape[1]] = result
            weights[i:i + target_size, col_begin:x.shape[1]] += 1

        # 处理下方边缘数据
        row_begin = x.shape[0] - target_size
        for i in tqdm(range(0, x.shape[1] - target_size, target_size)):
            img_one = x[row_begin:x.shape[0], i:i + target_size, :]
            pre_one = predict(img_one)
            pre_one = pre_one.transpose(1, 2, 0)
            weight = weights[row_begin:x.shape[0], i:i + target_size]
            pre_current = pad_y[row_begin:x.shape[0], i:i + target_size]
            result = (weight * pre_current + pre_one) * (1 / (weight + 1))
            pad_y[row_begin:x.shape[0], i:i + target_size] = result
            weights[row_begin:x.shape[0], i:i + target_size] += 1
        
        # 处理右下角数据
        img_one = x[x.shape[0] - target_size:x.shape[0], x.shape[1] - target_size:x.shape[1], :]
        pre_one = predict(img_one)
        pre_one = pre_one.transpose(1, 2, 0)
        weight = weights[x.shape[0] - target_size:x.shape[0], x.shape[1] - target_size:x.shape[1]]
        pre_current = pad_y[x.shape[0] - target_size:x.shape[0], x.shape[1] - target_size:x.shape[1]]
        result = (weight * pre_current + pre_one) * (1 / (weight + 1))
        pad_y[x.shape[0] - target_size:x.shape[0], x.shape[1] - target_size:x.shape[1]] = result
        weights[x.shape[0] - target_size:x.shape[0], x.shape[1] - target_size:x.shape[1]] += 1

        return pad_y


    def main(self, allpath, outpath, solver, overlap_rate=0.5, target_size=256, totif=False, class_num=0):  # 模型，所有图片路径列表，输出图片路径
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            dataset = gdal.Open(one_path)
            if dataset == None:
                print("failed to open img")
                sys.exit(1)
            pic = dataset.ReadAsArray()
            pic = pic.transpose(1,2,0)
            pic = pic.astype(np.float32)

            y_probs = self.make_prediction_wHy(x=pic, target_size=target_size, overlap_rate=overlap_rate, predict = lambda xx: solver.predict_x(xx), class_num=class_num) # 数据，目标大小，重叠度 预测函数 预测类别数，返回每次识别的

            y_ori = np.argmax(y_probs, axis=2)
            d, n = os.path.split(one_path)

            if totif:
                self.CreatTf(one_path, y_ori, outpath)

            img_out = np.zeros(y_ori.shape + (3,))
            img_out = img_out.astype(np.int16)
            for i in range(self.class_number):
                img_out[y_ori == i, :] = COLOR_DICT[i]  # 对应上色
            save_file = os.path.join(outpath, n[:-4] + '_color' + '.png')
            skimage.io.imsave(save_file, img_out)
            os.startfile(outpath)
            print('预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))


if __name__ == '__main__':


    predictImgPath = r'G:\manas_class\project_manas\0-src_img' # 待预测影像的文件夹路径
    Img_type = '*.dat' # 待预测影像的类型
    trainListRoot = r'G:\manas_class\project_manas\water\2-trainlist\trainlist_0727_balance_test.txt' #与模型训练相同的trainlist
    numclass = 2 # 样本类别数
    model = Dunet #模型
    model_path = r'G:\manas_class\project_manas\water\3-weights\Dunet-manans_water_balance_0728test.th' # 模型文件完整路径
    output_path = r'G:\manas_class\project_manas\water\3-predict_result_0728_balance_dunt_test_labelweight1.0' # 输出的预测结果路径
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

    #data_dict['mean'] = [92.663475, 97.823914, 90.74943] #自定义
    #data_dict['std'] = [44.311825, 41.875866, 38.67438] #自定义

    solver = SolverFrame(net = model(num_classes=numclass, band_num = band_num), data_dict=data_dict, band_num=band_num) 
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

    predict_instantiation = Predict(class_number = numclass)
    predict_instantiation.main(listpic, output_path, solver, target_size=target_size, overlap_rate=overlap_rate, totif = True, class_num = numclass)





