# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
Data Processing
数据处理
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import torch
import torch.utils.data as data
import skimage.io
import numpy as np
from tqdm import tqdm

class MyDataLoader(data.Dataset):
    def __init__(self,data_dict, root='', normalized_Label = False, band_num = 3):
        self.root = root
        self.normalized_Label = normalized_Label
        self.img_mean = data_dict['mean']
        self.std = data_dict['std']
        self.filelist = self.root
        self.band_num = band_num

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
        
        for i in range(self.band_num):
            img[:,:,i] -= self.img_mean[i]
        img = img / self.std

        img = img.transpose(2,0,1)
        if (self.normalized_Label == True): #标签根据需要做归一化
            label = label.transpose(2,0,1)/255.0
        else:
            label = label.transpose(2,0,1) 
        
        img = torch.Tensor(img)
        label = torch.Tensor(label)
        
        return img, label


class DataTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, classes_num = 2, trainlistPath="",
                 inform_data_file="", normVal=1.10, band_num = 3, label_norm = False, label_weight_scale_factor = 1):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
           band_num: The number of bands of the image
        """
        self.trainlistPath = trainlistPath
        self.classes = classes_num
        self.band_num = band_num
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(band_num, dtype=np.float32)
        self.std = np.zeros(band_num, dtype=np.float32)
        self.inform_data_file = inform_data_file
        self.label_norm = label_norm
        self.img_shape = (-1, -1, -1)
        self.label_weight_scale_factor = label_weight_scale_factor

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i])) # 平滑类别权重
            #self.classWeights[i] = 1 / (normHist[i] + 0.01) # 直接置倒数
        self.classWeights = np.power(self.classWeights, self.label_weight_scale_factor) # 根据标签权重系数缩放

    def readWholeTrainSet(self, trainlistPath, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0

        with open(trainlistPath, 'r') as f:
            textFile = f.readlines()
            
            img_file, label_file = textFile[0].split()
            self.img_shape = np.shape(skimage.io.imread(img_file))

            for line in tqdm(textFile):
                img_file, label_file = line.split()
                img_data = skimage.io.imread(img_file)
                label_data = skimage.io.imread(label_file, as_gray=True)
                
                if self.label_norm == True:
                    label_data = label_data/255

                unique_values = np.unique(label_data)

                max_unique_value = max(unique_values)
                min_unique_value = min(unique_values)

                if train_flag == True:
                    hist = np.histogram(label_data, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    for i in range(self.band_num):
                        self.mean[i] += np.mean(img_data[:, :, i])
                        self.std[i] += np.std(img_data[:, :, i])

                if max_unique_value > (self.classes - 1) or min_unique_value < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(trainlistPath=self.trainlistPath)

        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            data_dict['img_shape'] = self.img_shape
            return data_dict
        return None
          

        