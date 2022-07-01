import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import os
from time import time
from networks.dinknet import  DinkNet34, DinkNet101   
from networks.unet import Unet
from networks.dunet import Dunet
from networks.deeplabv3 import DeepLabv3_plus, ResNet
from networks.fcn8s import FCN8S
from framework import MyFrame
from loss import CrossEntropyLoss2d
from data import DataLoader_wHy
import tqdm
import numpy as np

SHAPE = (256,256) #数据维度

trainListRoot = r'E:\GID_test\2-trainlist\trainlist_0629.txt' #训练样本列表
numclass = 6
NAME = 'DinkNet101-GIDTest' # 训练模型保存名     
model = DinkNet101 #选择的训练模型

loss = CrossEntropyLoss2d
#loss = dice_bce_loss
solver = MyFrame(net=model(num_classes=numclass), loss=loss, lr=1e-5) # 网络，损失函数，以及学习率
modelFilesSavePath = 'weights/' + NAME + '.th'
if os.path.exists(modelFilesSavePath):
    solver.load(modelFilesSavePath)
    print('继续训练')
batchsize = 8#计算批次大小
init_lr = 1e-2

dataset = DataLoader_wHy(root = trainListRoot, normalized_Label=False) #读取数据

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)#多进程

mylog = open('logs/'+NAME+'.log','w')#日志文件
tic = time()
no_optim = 0
total_epoch = 3000 #训练次数
train_epoch_best_loss = 100. #预期结果

print('Is cuda availabel: ', torch.cuda.is_available())
print('Cuda device count: ', torch.cuda.device_count())
print('Current device: ', torch.cuda.current_device())

for epoch in tqdm.tqdm(range(1, total_epoch + 1)):
    data_loader_iter = iter(data_loader)#迭代器
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()#优化器
        train_epoch_loss += train_loss
        print(train_loss,'---', train_epoch_loss, 'len: ', len(data_loader_iter))
    train_epoch_loss /= len(data_loader_iter)

    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:#保留最好的loss
        no_optim += 1
    else:#小于最好的
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss #保留结果
        solver.save(modelFilesSavePath)
    if no_optim > 30:
        print(mylog, 'early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 15:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
       #solver.update_lr(init_lr, epoch, total_epoch)
    mylog.flush()

print('Finish!')
mylog.close()