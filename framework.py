# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
Training Framework and Functions
训练框架及函数
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)#优化器

        self.loss = loss
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
    
        
    def forward(self, volatile=False): # 添加变量
        self.img = Variable(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = Variable(self.mask.long().cuda(), volatile=volatile)
        
    def optimize(self, ifStep=True):
        self.forward()
        pred = self.net.forward(self.img) # 网络中输入数据
        label = self.mask.cpu().squeeze().cuda()
        loss = self.loss(output = pred, target = label)
        loss.backward() # 反向传播梯度
        if ifStep:
            self.optimizer.step() # 更新所有参数
            self.optimizer.zero_grad() # 清空梯度
        return loss.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr_geometric_decline(self, new_lr, mylog, factor=False): # 学习率等比下降更新
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        mylog.write('update learning rate: %f -> %f' % (self.old_lr, new_lr) + '\n')

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
    
    def update_lr_standard(self, init_lr, now_it, total_it, mylog): # 学习率标准下降更新
        power = 0.9
        lr = init_lr * (1 - float(now_it) / total_it) ** power
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        mylog.write('update learning rate: %f -> %f' % (self.old_lr, lr) + '\n')
        print('update learning rate: %f -> %f' % (self.old_lr, lr))

        self.old_lr = lr
