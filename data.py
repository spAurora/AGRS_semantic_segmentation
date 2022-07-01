"""

"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import skimage.io
from PIL import Image
import pickle
import cv2
import numpy as np
import os

class DataLoader(data.Dataset):
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

    def __init__(self, classes_num = 6, trainlistPath="",
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.trainlistPath = trainlistPath
        self.classes = classes_num
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, trainlistPath='', train_flag=True):
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
            for line in textFile:
                img_file, label_file = line.split()
                img_data = skimage.io.imread(img_file)
                label_data = skimage.io.imread(label_file, as_gray=True)
                
                unique_values = np.unique(label_data)

                max_unique_value = max(unique_values)
                min_unique_value = min(unique_values)

                if train_flag == True:
                    hist = np.histogram(label_data, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    self.mean[0] += np.mean(img_data[:, :, 0])
                    self.mean[1] += np.mean(img_data[:, :, 1])
                    self.mean[2] += np.mean(img_data[:, :, 2])

                    self.std[0] += np.std(img_data[:, :, 0])
                    self.std[1] += np.std(img_data[:, :, 1])
                    self.std[2] += np.std(img_data[:, :, 2])

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

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            return data_dict
        return None
          

        