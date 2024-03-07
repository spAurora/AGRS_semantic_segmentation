"""
对一个目录下的所有影像进行下采样, 转移到一个新的目录里
~~~~~~~~~~~~~~~~
code by Zhang Chi
Aerospace Information Research Institute, Chinese Academy of Sciences
University of Chinese Academy of Sciences
yiguanxianyu@gmail.com
zhangchi233@mails.ucas.ac.cn
"""
import os
import subprocess

ratio = 0.2
src_folder = "/Users/xianyu/code/random/ToZC"  # 源文件夹路径
dest_folder = "/Users/xianyu/code/random/output"  # 目标文件夹路径

ratio_percentage = f"{int(ratio*100)}%"

for root, dirs, files in os.walk(src_folder):
    for file in files:
        if file.endswith(".tif"):
            input_file_path = os.path.join(root, file)

            # 构建目标路径
            relative_path = os.path.relpath(root, src_folder)
            target_dir = os.path.join(dest_folder, relative_path)
            output_file_path = os.path.join(target_dir, file)

            os.makedirs(target_dir, exist_ok=True)

            # 调用process_tif处理文件，并将结果保存到目标路径
            subprocess.run(
                [
                    "gdal_translate",
                    "-outsize",
                    ratio_percentage,
                    ratio_percentage,
                    "-r",
                    "bilinear",
                    input_file_path,
                    output_file_path,
                ]
            )
