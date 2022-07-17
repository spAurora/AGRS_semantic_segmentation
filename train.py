import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import os
from time import time
from networks.dinknet import  DLinkNet34, DLinkNet50, DLinkNet101   
from networks.unet import Unet
from networks.dunet import Dunet
from networks.deeplabv3 import DeepLabv3_plus, ResNet
from networks.fcn8s import FCN8S
from framework import MyFrame
from loss import CrossEntropyLoss2d, FocalLoss2d
from data import DataLoader, DataTrainInform
from tqdm import tqdm
import numpy as np

SHAPE = (256,256) #数据维度

trainListRoot = r'E:\manas_class\project_manas\glacier\2-trainlist\trainlist_0713.txt' #训练样本列表
save_model_path = r'E:\manas_class\project_manas\glacier\3-weights' #训练模型保存路径  
model = DLinkNet34 #选择的训练模型
save_model_name = 'DinkNet34-manans_glacier.th' #训练模型保存名   
loss = FocalLoss2d #损失函数
numclass = 2 #样本类别数
batchsize = 8 #计算批次大小
init_lr = 0.005 #初始学习率
total_epoch = 300 #训练次数
band_num = 4 #影像的波段数
label_norm = True # 是否对标签进行归一化 针对0/255二分类标签

mylog = open('logs/'+save_model_name[:-3]+'.log', 'w') #日志文件

print('Is cuda availabel: ', torch.cuda.is_available())
print('Cuda device count: ', torch.cuda.device_count())
print('Current device: ', torch.cuda.current_device())

dataCollect = DataTrainInform(classes_num=numclass, trainlistPath=trainListRoot, band_num=band_num, label_norm=label_norm) #计算数据集信息
data_dict = dataCollect.collectDataAndSave()
if data_dict is None:
    print("error while pickling data. Please check.")
    exit(-1)
print('data mean:', data_dict['mean'])
print('data std: ', data_dict['std'])
print('label weight: ', data_dict['classWeights']) 

weight = torch.from_numpy(data_dict['classWeights']).cuda()
loss = loss(weight=weight) #loss实例化

solver = MyFrame(net=model(num_classes=numclass, band_num = band_num), loss=loss, lr=init_lr) #网络，损失函数，以及学习率
save_model_full_path = save_model_path + '/' + save_model_name
if os.path.exists(save_model_full_path):
    solver.load(save_model_full_path)
    print('继续训练')

dataset = DataLoader(root = trainListRoot, normalized_Label=label_norm, data_dict=data_dict, band_num=band_num) #读取训练集

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0) #定义训练数据装载器

print('size ', len(dataset)/batchsize)

#初始化最佳loss和未优化epoch轮数
train_epoch_best_loss = 100 
no_optim = 0
tic = time()

for epoch in tqdm(range(1, total_epoch + 1)):
    
    data_loader_iter = iter(data_loader) #迭代器
    train_epoch_loss = 0
    for img, mask in tqdm(data_loader_iter):
        solver.set_input(img, mask)
        train_loss = solver.optimize() #优化器
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)

    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    print(solver.optimizer.state_dict()['param_groups'][0]['lr'])
    
    if train_epoch_loss >= train_epoch_best_loss:#保留最好的loss
        no_optim += 1
    else:#小于最好的
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss #保留结果
        solver.save(save_model_full_path)
    if no_optim > 12:
        print(mylog, 'early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 0:
        if solver.old_lr < 5e-7:
            break
        solver.load(save_model_full_path)
        solver.update_lr(2.0, factor = True, mylog = mylog)
    mylog.flush()

print('Finish!')
mylog.close()