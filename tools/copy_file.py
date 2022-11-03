# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
复制文件，用于检索img对应的label
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

from shutil import copyfile
import fnmatch
import os

img_path = r'C:\Users\75198\OneDrive\论文\中文5-供稿2\图\样本集示例\img' # 存储原始影像文件夹
label_path = r'E:\project_UAV\2-enhance_label' # 原标签文件夹
label_target_path = r'C:\Users\75198\OneDrive\论文\中文5-供稿2\图\样本集示例\label' # 目标标签文件夹

img_list = fnmatch.filter(os.listdir(img_path), '*.tif') # 过滤出所有tif文件

for img_file in img_list:
    label_full_path = label_path + '/' + img_file
    label_target_full_path = label_target_path + '/' + img_file
    copyfile(label_full_path, label_target_full_path)