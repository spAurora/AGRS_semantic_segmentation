# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
Model Training
模型训练
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import torch
import os
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
from torchsummary import summary

from framework import MyFrame
from loss import CrossEntropyLoss2d, FocalLoss2d
from data import MyDataLoader, DataTrainInform
from test import GetTestIndicator

from networks.DLinknet import DLinkNet34, DLinkNet50, DLinkNet101
from networks.UNet_Light import UNet_Light
from networks.UNet import UNet
from networks.DUNet_Plus import DUNet_Plus
from networks.Deeplab_v3_plus import DeepLabv3_plus
from networks.FCN8S import FCN8S
from networks.DABNet import DABNet
from networks.Segformer import Segformer
from networks.RS_Segformer import RS_Segformer
from networks.DE_Segformer import DE_Segformer
from networks.UNetFormer import UNetFormer
from networks.HRNet import HRNet
from networks.FCN import FCN_ResNet50, FCN_ResNet101
from networks.U_MobileNet import U_MobileNet
from networks.SegNet import SegNet
from networks.U_ConvNeXt import U_ConvNeXt
from networks.ViT import ViTSegmentation
# from networks.U_ConvNeXt_HWD import U_ConvNeXt_HWD
# from networks.U_ConvNeXt_HWD_DS import U_ConvNeXt_HWD_DS

from networks.MAE_Seg import MAEViTSegmentation

'''参数设置'''
trainListRoot = r'E:\project_global_populus\MAE_test_250324\2-train_list\trainlist-250324_0.2.txt'  # 训练样本列表
save_model_path = r'E:\project_global_populus\MAE_test_250324\3-weights'  # 训练模型保存路径
model = MAEViTSegmentation  # 选择的训练模型
save_model_name = 'MAE_TF_test_FPN.pth'  # 训练模型保存名
mylog = open('logs/'+save_model_name[:-4]+'.log', 'w')  # 日志文件
loss = FocalLoss2d  # 损失函数
classes_num = 2  # 样本类别数
batch_size = 8  # 计算批次大小
init_lr = 0.0001  # 初始学习率
total_epoch = 300  # 训练次数
band_num = 4  # 影像的波段数
if_norm_label = True  # 是否对标签进行归一化 0/255二分类应设置为True
label_weight_scale_factor = 1  # 标签权重的指数缩放系数 1为不缩放

if_vis = False  # 是否输出中间可视化信息 一般设置为False，设置为True需要模型支持
if_open_profile = False  # 是否启用性能分析，启用后计算2个eopch即终止训练并打印报告，仅供硬件负载分析和性能优化使用

lr_mode = 0  # 学习率更新模式，0为等比下降，1为标准下降
max_no_optim_num = 1  # 最大loss无优化次数
lr_update_rate = 3.0  # 学习率等比下降更新率
min_lr = 1e-6  # 最低学习率

simulate_batch_size = True  # 是否模拟大batchsize；除非显存太小一般不开启
simulate_batch_size_num = 8 # 模拟batchsize倍数 最终batchsize = simulate_batch_size_num * batch_size

full_cpu_mode = True  # 是否全负荷使用CPU，默认pytroch使用cpu一半核心

if_open_test = True  # 是否开启测试模式
test_img_path = r'E:\project_global_populus\MAE_test_250324\4-predict_result\test_img'  # 测试集影像文件夹
test_label_path = r'E:\project_global_populus\MAE_test_250324\4-predict_result\test_label'  # 测试集真值标签文件夹
test_output_path = r'E:\project_global_populus\MAE_test_250324\4-predict_result\temp'
target_size = 224  # 模型预测窗口大小，与训练模型一致
test_img_type = '*.tif'  # 测试集影像数据类型

if_MAE_finetune = True # 是否为MAE微调模式 

if_print_model_summary = True
if model.__name__ in ['HRNet', 'FCN_ResNet50', 'FCN_ResNet101', 'SegNet', 'U_ConvNeXt_HWD', 'U_ConvNeXt_HWD_DS', 'MAEViTSegmentation', 'ViTSegmentation']:  # 是否输出模型参数信息 部分模型不可用
    if_print_model_summary = False
else:
    pass

if if_open_test:
    if not os.path.exists(test_output_path):
        os.mkdir(test_output_path)

'''全负荷使用CPU'''
if full_cpu_mode:
    cpu_num = cpu_count()  # 自动获取最大核心数目
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

'''收集系统环境信息'''
tic = time.time()
format_time = time.asctime(time.localtime(tic))  # 系统当前时间
print(format_time)
mylog.write(format_time + '\n')

print('Using cpu core num: ', cpu_num)
print('Is cuda availabel: ', torch.cuda.is_available())  # 是否支持cuda
print('Cuda device count: ', torch.cuda.device_count())  # 显卡数
print('Current device: ', torch.cuda.current_device())  # 当前计算的显卡id

'''收集数据集信息'''
dataCollect = DataTrainInform(classes_num=classes_num, trainlistPath=trainListRoot, band_num=band_num,
                              label_norm=if_norm_label, label_weight_scale_factor=label_weight_scale_factor)  # 计算数据集信息
data_dict = dataCollect.collectDataAndSave()  # 数据集信息存储于字典中
# '''手动设置data_dict'''
# data_dict = {}
# data_dict['mean'] = [49.017967, 49.88055, 50.7376, 64.34752]
# data_dict['std'] = [5.3343625, 7.126632, 8.821242, 8.598516]
# data_dict['classWeights'] = np.array([1.4093286, 6.478462], dtype=np.float32)
# data_dict['img_shape'] = [256, 256, 4]

if data_dict is None:
    print("error while pickling data. Please check.")
    exit(-1)
print('data mean:', data_dict['mean'])
print('data std: ', data_dict['std'])
print('label weight: ', data_dict['classWeights'])
print('img shape: ', data_dict['img_shape'])
mylog.write('data_dict: ' + str(data_dict) + '\n')
mylog.flush()

'''初始化solver'''
weight = torch.from_numpy(data_dict['classWeights']).cuda()
loss = loss(weight=weight)

if if_vis:
    solver = MyFrame(net=model(num_classes=classes_num, band_num=band_num,
                     ifVis=if_vis), loss=loss, lr=init_lr)  # 初始化网络，损失函数，学习率
else:
    solver = MyFrame(net=model(num_classes=classes_num,
                     band_num=band_num), loss=loss, lr=init_lr)  # 初始化网络，损失函数，学习率

save_model_full_path = save_model_path + '/' + save_model_name
if os.path.exists(save_model_full_path):
    solver.load(save_model_full_path, if_MAE_finetune)
    print('---------\n***Resume Training***\n---------')
else:
    print('---------\n***New Training***\n---------')

'''输出模型信息'''
torch_shape = np.array(data_dict['img_shape'])
torch_shape = [torch_shape[2], torch_shape[0], torch_shape[1]]
if if_print_model_summary is True:
    summary((solver.net), input_size=tuple(torch_shape), batch_size=batch_size)


'''初始化dataloader'''
dataset = MyDataLoader(root=trainListRoot, normalized_Label=if_norm_label,
                       data_dict=data_dict, band_num=band_num)  # 读取训练数据集
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  # 定义训练数据装载器 开启锁页内存
print('Number of Iterations: ', int(len(dataset)/batch_size))

'''模型训练'''
# 初始化最佳loss和未优化epoch轮数
train_epoch_best_loss = 100
no_optim = 0
print('-------------------------------------------')
with torch.autograd.profiler.profile(enabled=if_open_profile, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    for epoch in tqdm(range(1, total_epoch + 1)):
        data_loader_iter = iter(data_loader)  # 初始化迭代器
        train_epoch_loss = 0
        cnt = 0
        for img, mask in tqdm(data_loader_iter):
            cnt = cnt + 1  # 计数累加
            solver.set_input(img, mask)  # 设置batch的影像和标签输入
            if simulate_batch_size:
                if (cnt % simulate_batch_size_num == 0):  # 模拟大batchsize
                    train_loss = solver.optimize(ifStep=True, ifVis=if_vis)
                else:
                    train_loss = solver.optimize(ifStep=False, ifVis=if_vis)
            else:
                train_loss = solver.optimize(
                    ifStep=True, ifVis=if_vis)  # 非模拟大batchsize，每次迭代都更新参数
            train_epoch_loss += train_loss
            mylog.write('epoch: %d iter: %d train_iter_loss: %f learn_rate: %f ' % (
                epoch, cnt, train_loss, solver.old_lr) + '\n')  # 打印日志
        train_epoch_loss /= len(data_loader_iter)  # 计算该epoch的平均loss

        if if_open_test:  # 如果开启测试模型就在测试集上计算精度指标
            p, r, f = GetTestIndicator(net=solver.net, data_dict=data_dict, target_size=target_size, band_num=band_num, img_type=test_img_type,
                                       test_img_path=test_img_path, test_label_path=test_label_path, if_norm_label=if_norm_label, test_output_path=test_output_path)

        print('\nepoch:', epoch, '  training time:', int(time.time()-tic), 's')
        print('epoch average train loss:', train_epoch_loss)
        if if_open_test:
            print('epoch test indicator: precision=' + str(p) +
                  ', recall=' + str(r) + ', f1_score=' + str(f))
        print('current learn rate: ', solver.optimizer.state_dict()
              ['param_groups'][0]['lr'])
        print('---------')

        if if_open_test:
            mylog.write('epoch: %d train_epoch_loss: %f learn_rate: %f test_p: %f test_r: %f test_f: %f' % (
                epoch, train_epoch_loss, solver.old_lr, p, r, f) + '\n')
        else:
            mylog.write('epoch: %d train_epoch_loss: %f learn_rate: %f ' % (
                epoch, train_epoch_loss, solver.old_lr) + '\n')  # 打印日志

        if lr_mode == 0:
            if train_epoch_loss >= train_epoch_best_loss:  # 若当前epoch的loss大于等于之前最小的loss
                no_optim += 1
            else:  # 若当前epoch的loss小于之前最小的loss
                no_optim = 0  # loss未降低的轮数归0
                train_epoch_best_loss = train_epoch_loss  # 保留当前epoch的loss
                solver.save(save_model_full_path)  # 保留当前epoch的模型
            if no_optim > 9:  # 若过多epoch后loss仍不下降则终止训练
                print(mylog, 'early stop at %d epoch' % epoch)  # 打印信息至日志
                print('early stop at %d epoch' % epoch)
                break
            if no_optim > max_no_optim_num:  # 多轮epoch后loss不下降则更新学习率
                if solver.old_lr < min_lr:  # 当前学习率过低终止训练
                    break
                solver.load(save_model_full_path, if_MAE_finetune)  # 读取保存的loss最低的模型
                solver.update_lr_geometric_decline(
                    lr_update_rate, factor=True, mylog=mylog)  # 更新学习率
                no_optim = 0  # loss未降低轮数归0
        elif lr_mode == 1:
            if train_epoch_loss >= train_epoch_best_loss:
                train_epoch_best_loss = train_epoch_loss
                solver.save(save_model_full_path)
            solver.update_lr_standard(
                init_lr=init_lr, now_it=epoch, total_it=total_epoch+1, mylog=mylog)

        mylog.flush()

        if if_open_profile:
            if epoch >= 2:
                print('training break beacuse if_open_profile==True')
                break  # 性能分析模式仅计算2个epoch

if if_open_profile:
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # prof.export_chrome_trace('profile.json')

print('\n---------')
print('Training completed')
mylog.close()
