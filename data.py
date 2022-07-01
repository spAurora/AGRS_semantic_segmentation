"""

"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import skimage.io
from PIL import Image

import cv2
import numpy as np
import os

class DataLoader_wHy(data.Dataset):
    def __init__(self, root='', normalized_Label = False):
        self.root = root
        self.normalized_Label = normalized_Label

        self.filelist = self.root

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines() # 返回一个列表，其中包含文件中的每一行作为列表项
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, label_file = self.filelist[index].split()

        img = skimage.io.imread(img_file)
        label = skimage.io.imread(label_file, as_gray=True)
        
        img = np.array(img, np.float32)
        label = np.array(label, np.float32)

        label = np.expand_dims(label, axis=2) #标签增加一个维度 (H W C)

        img = img.transpose(2,0,1)/255.0 * 3.2 - 1.6 #影像归一化
        if (self.normalized_Label == False): #标签根据需要做归一化
            label = label.transpose(2,0,1) 
        else:
            label = label.transpose(2,0,1)/255.0
        
        img = torch.Tensor(img)
        label = torch.Tensor(label)
        
        return img, label
            
            

        