#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
扩充影像
随机裁剪、六方向旋转
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import numpy as np
import os
import fnmatch
from tqdm import tqdm
import gdal
import sys
from skimage import transform
from PIL import Image
from noise import pnoise2, snoise2
import math

def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    del im_dataset

    return im_data


def write_img(out_path, im_data, mode=1, rotate=0):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # 生成随机种子
    seed = np.random.randint(0, 100)
    # identify data type
    if mode == 0:
        datatype = gdal.GDT_Byte
    else:
        datatype = gdal.GDT_Byte

    # calculate number of bands
    if len(im_data.shape) > 2:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)

    for i in range(im_bands):
        if mode == 0:
            tmp = im_data
        elif mode == 1:
            tmp = im_data[i]
        else:
            print('mode should 0 or 1!')
        im = Image.fromarray(tmp)

        if rotate == 0:
            out = im
        elif rotate == 1:
            out = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif rotate == 2:
            out = im.transpose(Image.FLIP_TOP_BOTTOM)
        elif rotate == 3:
            out = im.transpose(Image.ROTATE_90)
        elif rotate == 4:
            out = im.transpose(Image.ROTATE_180)
        elif rotate == 5:
            out = im.transpose(Image.ROTATE_270)

        tmp = np.array(out)

        new_dataset.GetRasterBand(i + 1).WriteArray(tmp)

    del new_dataset


images_path = r'C:\Users\75198\OneDrive\project_paper_3\0-other_data\real_haze'  # 原始影像路径 栅格
save_img_path = r'D:\github_repository\PyTorch-CycleGAN\datasets\syn2real\train\B'  # 保存增强后影像路径

expandNum = 4  # 每个样本的基础扩充数目，最终数目会在基础扩充数目上*6
randomCorpSize = 256  # 随机裁剪后的样本大小
img_edge_width = 512  # 输入影像的大小

max_thread = randomCorpSize / img_edge_width

image_list = fnmatch.filter(os.listdir(images_path), '*.tif')  # 过滤出tif文件

for img_name in tqdm(image_list):
    img_full_path = os.path.join(images_path + '/' + img_name)

    '''读取img和label数据'''
    sr_img = read_img(img_full_path)

    sr_img = sr_img.transpose(1, 2, 0)

    '''样本扩增'''
    cnt = 0
    for i in range(expandNum):

        p1 = np.random.choice([0, max_thread])  # 最大height比例
        p2 = np.random.choice([0, max_thread])  # 最大width比例

        start_x = int(p1 * img_edge_width)
        start_y = int(p2 * img_edge_width)

        new_sr_img = sr_img[start_x:start_x + randomCorpSize,
                            start_y:start_y + randomCorpSize, :]

        new_sr_img = new_sr_img.transpose(2, 0, 1)
        for j in range(6):
            save_img_full_path = save_img_path + '/' + \
                img_name[0:-4] + '_' + str(cnt) + '.tif'
            cnt += 1

            write_img(save_img_full_path, new_sr_img,
                      mode=1, rotate=j)
