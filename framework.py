# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
Training Framework and Functions
训练框架及函数
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import torch
import torch.nn as nn
import torch.amp as amp

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net.cuda() # 启动模型GPU计算
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count())) # 启动多卡并行训练
        # self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr) # 采用Adam优化器
        self.optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=lr, weight_decay=0.001)
        self.loss = loss # 初始化损失函数
        self.old_lr = lr # 初始化学习率
        self.scaler = amp.GradScaler()  # Initializing GradScaler for mixed precision training

        if evalmode: # 测试模式
            for i in self.net.modules(): # 返回网络所有的元素，包括不同级别的子元素
                if isinstance(i, nn.BatchNorm2d): # 返回判断对象的变量类型匹配结果
                    i.eval() # 设置BN层单独失活
        
    def set_input(self, img_batch, mask_batch=None): # 设置每一个batch的img和label输入
        self.img = img_batch
        self.mask = mask_batch

    @amp.autocast('cuda')  # Autocasting for mixed precision
    def optimize(self, ifStep=True, ifVis = False):
        self.img = self.img.cuda(non_blocking=True)
        if self.mask is not None:
            self.mask = self.mask.cuda(non_blocking=True).to(torch.int64)

        # ifVis在这里没有任何用处，前向传递计算输出
        if ifVis: # 带可视化输出
            pred, _ = self.net(self.img)
        else:
            pred = self.net(self.img) 

        label = self.mask.squeeze() # 注意此处label维度规整可能会引发异常
        loss = self.loss(output=pred, target=label) # 计算loss
        self.scaler.scale(loss).backward()  # Scaling the loss before backward pass
        
        if ifStep:
            self.scaler.step(self.optimizer)  # Scale optimizer step
            self.scaler.update()  # Update the scaler for next iteration
            self.optimizer.zero_grad(set_to_none=True)  # Clear gradients
        
        return loss.item() # 返回loss张量的值
        
    def save(self, path):
        torch.save(self.net.state_dict(), path) # 模型保存
        
    def load(self, path, if_MAE_finetune):
        if if_MAE_finetune:
            self.net.module.load_state_dict(torch.load(path), strict=False) # MAE微调仅读取编码器
            self.net.module.freeze_encoder() # 冻结编码器权重
        else:
            self.net.load_state_dict(torch.load(path)) # 模型读取

    def update_lr_geometric_decline(self, rate, mylog, factor=False, log_print = False): # 学习率等比下降更新
        if factor:
            new_lr = self.old_lr / rate # 等比更新
        for param_group in self.optimizer.param_groups: # 参数组中记录当前学习率
            param_group['lr'] = new_lr

        if log_print:
            mylog.write('update learning rate: %f -> %f' % (self.old_lr, new_lr) + '\n') # 打印日志

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr)) # 终端信息显示输出
        
        self.old_lr = new_lr
    
    def update_lr_standard(self, init_lr, now_it, total_it, mylog, log_print = False): # 学习率标准下降更新
        power = 0.9
        lr = init_lr * (1 - float(now_it) / total_it) ** power
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        if log_print:
            mylog.write('update learning rate: %f -> %f' % (self.old_lr, lr) + '\n')
        
        print('update learning rate: %f -> %f' % (self.old_lr, lr))

        self.old_lr = lr
