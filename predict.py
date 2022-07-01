import skimage.io
import numpy as np
import os
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import glob
import torch
from networks.dinknet import DinkNet34, DinkNet101
from networks.unet import Unet
from networks.dunet import Dunet
from networks.deeplabv3 import DeepLabv3_plus
from networks.fcn8s import FCN8S
from torch.autograd import Variable as V
from PIL import Image
import cv2
import fnmatch
from PIL import Image
from data import DataTrainInform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCHSIZE_PER_CARD = 4
background = [0, 0, 0]
built_up = [255, 0, 0]
farmland = [0, 255, 0]
forest = [0, 255, 255]
meadow = [255, 255, 0]
water = [0, 0, 255]
COLOR_DICT = np.array([background, built_up, farmland, forest, meadow, water]) 

farmland = [255, 255, 255]
non_farmland = [0, 0, 0]
COLOR_DICT = np.array([non_farmland, farmland])

one_size = 256


# 在下文改了一下归一化！predict_x 与训练时一致 改了一下可以选择每个预测切片的大小
class TTAFrame():
    def __init__(self, net,data_dict, name='d34'):
        self.net = net.cuda()
        self.name = name
        self.img_mean = data_dict['mean']
        self.std = data_dict['std']


        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def predict_x(self, img):
        self.net.eval()

        img[:,:,0] -= self.img_mean[0]
        img[:,:,1] -= self.img_mean[1]
        img[:,:,2] -= self.img_mean[2]
        img = img / self.std

        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)
        img = V(torch.Tensor(img).cuda())
        maska = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)

        return maska

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

class P():
    def __init__(self, number):
        self.number = number


    def stretch(self, img):  # %2線性拉伸
        n = img.shape[2]
        for i in range(n):
            c1 = img[:, :, i]
            c = np.percentile(c1[c1 > 0], 2)  # 只拉伸大于零的值
            d = np.percentile(c1[c1 > 0], 98)
            t = (img[:, :, i] - c) / (d - c)
            t *= 65535
            t[t < 0] = 0
            t[t > 65535] = 65535
            img[:, :, i] = t
        return img

    def CreatTf(self, file_path_img, data, outpath, type=1):  # 原始文件，识别后的文件数组形式，新保存文件 1type为二值化 0为原始 区别在于文件名
        print(file_path_img)
        d, n = os.path.split(file_path_img)
        dataset = gdal.Open(file_path_img, GA_ReadOnly)  # 打开图片只读

        projinfo = dataset.GetProjection()  # 获取坐标系
        geotransform = dataset.GetGeoTransform()

        format = "GTiff"
        driver = gdal.GetDriverByName(format)  # 数据格式
        if type == 1:
            name = n[:-4] + '_result' + '.tif'  # 输出文件名字
        else:
            name = n[:-4] + '_ori_result' + '.tif'

        dst_ds = driver.Create(os.path.join(outpath, name), dataset.RasterXSize, dataset.RasterYSize,
                               1, gdal.GDT_Byte)  # 创建一个新的文件
        dst_ds.SetGeoTransform(geotransform)  # 投影
        dst_ds.SetProjection(projinfo)  # 坐标
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.FlushCache()
    
    def make_prediction_wHy(self, x, target_size, overlap, predict, class_num):
        weights = np.zeros((x.shape[0], x.shape[1], class_num), dtype=np.float32)
        space = int(target_size * (1-overlap))
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


    def main_p(self, allpath, maskpath, outpath, fun, rate=0.5, loss_cal=0, changes=False, totif=False, class_num=0):  # 模型，所有图片路径列表，输出图片路径
        print('执行预测...')
        num = 0
        for one_path in allpath:
            t0 = time.time()
            pic = skimage.io.imread(one_path)
            pic = pic.astype(np.float32)

            y_probs = self.make_prediction_wHy(pic, 256, 0.1, lambda xx: fun.predict_x(xx), class_num=class_num) # 数据，目标大小，重叠度 预测函数 预测类别数，返回每次识别的

            y_ori = np.argmax(y_probs, axis=2)
            d, n = os.path.split(one_path)

            if totif:
                self.CreatTf(one_path.replace('jpg','TIF'), y_ori, outpath, type=0)
            else:
                save_file = os.path.join(outpath,'/', n[:-4] + '_init' + '.png')
                skimage.io.imsave(save_file, y_ori)
                os.startfile(outpath)


            img_out = np.zeros(y_ori.shape + (3,))
            img_out = img_out.astype(np.int16)
            for i in range(self.number):
                img_out[y_ori == i, :] = COLOR_DICT[i]  # 对应上色
            save_file = os.path.join(outpath, n[:-4] + '_color' + '.png')
            skimage.io.imsave(save_file, img_out)
            os.startfile(outpath)
            print('预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))
            num += 1


if __name__ == '__main__':


    predictImgPath = r'D:\AGRS\results_why\test_image'
    trainListRoot = r'E:\FarmLandDataset\2-trainlist\trainlist_0701_small.txt'
    numclass = 2
    model = DinkNet34

    dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot) #计算数据集信息
    data_dict = dataCollect.collectDataAndSave()

    solver = TTAFrame(net = model(num_classes=numclass), name='dlink34', data_dict=data_dict)  # 根据批次识别类 
    solver.load(r'D:\AGRS/weights/DinkNet34-FarmLandTest.th')
    target = r'D:\AGRS\results_why\predict_result_farmlandTest'  #w 输出文件位置
    if not os.path.exists(target):
        os.mkdir(target)

    listpic = fnmatch.filter(os.listdir(predictImgPath), '*.jpg')
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    print(listpic)

    listmask = ['dataset/53.png', ]

    a = P(number = numclass)
    a.main_p(listpic, listmask, target, solver, rate = 0.5, loss_cal = 0, changes = False, totif = False, class_num = numclass) #w rate是二值化的比例





