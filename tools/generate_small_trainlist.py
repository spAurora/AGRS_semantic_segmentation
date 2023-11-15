#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
从训练列表里抽取部分生成小的训练列表
一般用于预训练
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

import random

def retain_lines(input_file, output_file, retain_percentage):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    num_lines_to_retain = int(total_lines * retain_percentage)

    # Randomly shuffle and retain lines
    random.shuffle(lines)
    retained_lines = lines[:num_lines_to_retain]

    with open(output_file, 'w') as f:
        f.writelines(retained_lines)

input_file_path = r'E:\project_daijiandi\2-trainlist\trainlist_231115.txt'
output_file_path = r'E:\project_daijiandi\2-trainlist\trainlist_231115_5pt.txt'
retain_percentage = 0.05 # retain percentage (e.g., 0.8 for 80% retention)

retain_lines(input_file_path, output_file_path, retain_percentage)