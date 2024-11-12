#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
删除空白图像or标签
！！！标签路径和图像路径可以颠倒，适合不同情况！！！
~~~~~~~~~~~~~~~~
code by ZC
Aerospace Information Research Institute, Chinese Academy of Sciences
"""
from pathlib import Path
from random import random

import numpy as np
from PIL import Image

image_path = Path(r"E:\project_populus_UAV\2-enhance_label\1-pretrain_enhancelabel_ds025_240307")
label_path = Path(r"E:\project_populus_UAV\2-enhance_img\1-pretrain_enhanceimg_ds025_240307")

# 面积占比小于area_threshold的会被删除
area_threshold = 0.05
# 随机数小于prob_threshold的才会被删除，
# 即：如果prob_threshold设置为0.3，
# 有prob_threshold%的不合理图片会被保留
prob_threshold = 0.3

assert image_path.is_dir()

for x in image_path.iterdir():
    if not x.is_file():
        continue
    if not x.suffix == ".tif":
        continue

    img = Image.open(x)
    w, h = img.size
    black_rate = np.count_nonzero(img) / (w * h)
    img.close()

    if (black_rate < area_threshold) and (random() > area_threshold):
        x.unlink()

        label_file = label_path / x.name
        if label_file.is_file():
            label_file.unlink()