# %%
# -*- coding: utf-8 -*-
"""
可视化训练日志
~~~~~~~~~~~~~~~~
code by LC
Aerospace Information Research Institute, Chinese Academy of Sciences
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import scienceplots

# %%
log_path = r'D:\github\AGRS_semantic_segmentation\logs'
# matplotlib.rcParams['text.usetex'] = False
# plt.style.use(["science"])
for log in os.listdir(log_path):
    file = os.path.join(log_path, log)
    print(file)
    with open(file) as f:
        lines = f.readlines()
    loss, test_p = [], []
    for line in lines:
        if "train_iter_loss" in line:
            start = line.find("train_iter_loss") + len("train_iter_loss") + 2
            end = line.find("learn_rate")
            loss.append(float(line[start:end]))
        elif "test_p" in line:
            start = line.find("test_p") + len("test_p") + 2
            end = line.find("test_r")
            test_p.append(float(line[start:end]))
    # print(loss)
    # print(test_p)
    print('load done')
    # 开始绘图
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    # 两根y轴
    ax2 = ax.twinx()
    x_p = [23 * i for i in range(len(test_p))]
    # training loss
    ax.plot(loss, color="#05BFDB")
    # test_p，数据已经对齐
    ax2.plot(x_p, test_p, color="#FF6D60")
    labelFont = 14
    # 设置label
    ax.set_xlabel('Iter', fontsize=labelFont)
    ax.set_ylabel('Training Loss', color="#05BFDB", fontsize=labelFont)
    ax2.set_ylabel('Test-p', color='#FF6D60', fontsize=labelFont)
    plt.tight_layout()
    # 图片保存为同名的jpg
    plt.savefig(log[:-3] + "jpg", dpi=300)
    # plt.show()