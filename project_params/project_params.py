# -*- coding: utf-8 -*-

"""
该文件仅用来存储工程的参数配置
"""

'''
WV-2胡杨红柳分割 清晰数据集
'''
'''参数设置'''
trainListRoot = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-trainlist\8-trainlist_clear_240401.txt'  # 训练样本列表
save_model_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\3-weights'  # 训练模型保存路径
model = U_ConvNeXt  # 选择的训练模型
save_model_name = '8-U_ConvNeXt-huyang_clear_240513.th'  # 训练模型保存名
mylog = open('logs/'+save_model_name[:-3]+'.log', 'w')  # 日志文件
loss = FocalLoss2d  # 损失函数
classes_num = 3  # 样本类别数
batch_size = 4  # 计算批次大小
init_lr = 0.001  # 初始学习率
total_epoch = 300  # 训练次数
band_num = 3  # 影像的波段数
if_norm_label = False  # 是否对标签进行归一化 0/255二分类应设置为True
label_weight_scale_factor = 1  # 标签权重的指数缩放系数 1为不缩放

if_vis = False  # 是否输出中间可视化信息 一般设置为False，设置为True需要模型支持
if_open_profile = False  # 是否启用性能分析，启用后计算2个eopch即终止训练并打印报告，仅供硬件负载分析和性能优化使用

lr_mode = 0  # 学习率更新模式，0为等比下降，1为标准下降
max_no_optim_num = 1  # 最大loss无优化次数
lr_update_rate = 3.0  # 学习率等比下降更新率
min_lr = 1e-6  # 最低学习率

simulate_batch_size = False  # 是否模拟大batchsize；除非显存太小一般不开启
# 模拟batchsize倍数 最终batchsize = simulate_batch_size_num * batch_size
simulate_batch_size_num = 4

full_cpu_mode = True  # 是否全负荷使用CPU，默认pytroch使用cpu一半核心

if_open_test = True  # 是否开启测试模式
test_img_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-clip_img\1-clip_img_clear_for_clear_Evaluation_853'  # 测试集影像文件夹
test_label_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-raster_label\1-raster_label_clear_for_clear_Evaluation'  # 测试集真值标签文件夹
test_output_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\3-predict_result\0-test_temp'
target_size = 256  # 模型预测窗口大小，与训练模型一致
test_img_type = '*.tif'  # 测试集影像数据类型



'''
白大棚地膜
'''
'''参数设置'''
trainListRoot = r'E:\project_bai\2-trainlist\train_list_240523-1.txt'  # 训练样本列表
save_model_path = r'E:\project_bai\3-weights'  # 训练模型保存路径
model = UNetPlusPlus  # 选择的训练模型
save_model_name = 'UNetPlusPlus_240523-1.th'  # 训练模型保存名
mylog = open('logs/'+save_model_name[:-3]+'.log', 'w')  # 日志文件
loss = FocalLoss2d  # 损失函数
classes_num = 3  # 样本类别数
batch_size = 4  # 计算批次大小
init_lr = 0.001  # 初始学习率
total_epoch = 300  # 训练次数
band_num = 3  # 影像的波段数
if_norm_label = False  # 是否对标签进行归一化 0/255二分类应设置为True
label_weight_scale_factor = 1  # 标签权重的指数缩放系数 1为不缩放

if_vis = False  # 是否输出中间可视化信息 一般设置为False，设置为True需要模型支持
if_open_profile = False  # 是否启用性能分析，启用后计算2个eopch即终止训练并打印报告，仅供硬件负载分析和性能优化使用

lr_mode = 0  # 学习率更新模式，0为等比下降，1为标准下降
max_no_optim_num = 1  # 最大loss无优化次数
lr_update_rate = 3.0  # 学习率等比下降更新率
min_lr = 1e-6  # 最低学习率

simulate_batch_size = False  # 是否模拟大batchsize；除非显存太小一般不开启
# 模拟batchsize倍数 最终batchsize = simulate_batch_size_num * batch_size
simulate_batch_size_num = 4

full_cpu_mode = True  # 是否全负荷使用CPU，默认pytroch使用cpu一半核心

if_open_test = True  # 是否开启测试模式
test_img_path = r'E:\project_bai\0-test_img'  # 测试集影像文件夹
test_label_path = r'E:\project_bai\0-test_label'  # 测试集真值标签文件夹
test_output_path = r'E:\project_bai\4-predict_result\test_output'
target_size = 224  # 模型预测窗口大小，与训练模型一致
test_img_type = '*.png'  # 测试集影像数据类型

# 哈密隔壁砾幕层
'''参数设置'''
trainListRoot = r'E:\project_hami_limuceng\2-trainlist\train_list_240617.txt'  # 训练样本列表
save_model_path = r'E:\project_hami_limuceng\3-weights'  # 训练模型保存路径
model = U_ConvNeXt_HWD_DS  # 选择的训练模型
save_model_name = 'U_ConvNeXt_HWD_DS_240617.th'  # 训练模型保存名
mylog = open('logs/'+save_model_name[:-3]+'.log', 'w')  # 日志文件
loss = FocalLoss2d  # 损失函数
classes_num = 2  # 样本类别数
batch_size = 8  # 计算批次大小
init_lr = 0.001  # 初始学习率
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

simulate_batch_size = False  # 是否模拟大batchsize；除非显存太小一般不开启
# 模拟batchsize倍数 最终batchsize = simulate_batch_size_num * batch_size
simulate_batch_size_num = 4

full_cpu_mode = True  # 是否全负荷使用CPU，默认pytroch使用cpu一半核心

if_open_test = True  # 是否开启测试模式
test_img_path = r'E:\project_hami_limuceng\1-clip_img'  # 测试集影像文件夹
test_label_path = r'E:\project_hami_limuceng\1-raster_label'  # 测试集真值标签文件夹
test_output_path = r'E:\project_hami_limuceng\4-predict_result\0-test_temp'
target_size = 192  # 模型预测窗口大小，与训练模型一致
test_img_type = '*.tif'  # 测试集影像数据类型