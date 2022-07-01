from turtle import shape
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

import cv2
import numpy as np

class MyFrame():
    """
    一些参数函数
    """
    def __init__(self, net, loss, lr=2e-4, evalmode = False):#loss是一个函数
        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)#优化器

        self.loss = loss
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):#喂数据
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
    
        
    def forward(self, volatile=False):#添加变量
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.long().cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)#网络中输入数据
        label = self.mask.cpu().squeeze().cuda()
        loss = self.loss(output = pred, target = label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
