#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
均匀滑窗裁剪,仅图像
~~~~~~~~~~~~~~~~
code by ZC
Aerospace Information Research Institute, Chinese Academy of Sciences
"""


from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import os
import fnmatch


def cut_img(
    src_img: Path,
    output_img_dir: Path,
    output_file_name: str,
    block_size: int,
    overlap: float,
):

    overlap = int(overlap * block_size)
    src = rio.open(src_img)
    src_data = src.read()

    src_profile = src.profile
    src_profile.update(
        {
            "height": block_size,
            "width": block_size,
            "compress": "deflate",
        }
    )

    height, width = src.shape
    blocks = []
    for i in range(0, height, block_size - overlap):
        for j in range(0, width, block_size - overlap):
            if height - i < block_size: # 右侧及下侧
                i = height - block_size
            if width - j < block_size:
                j = width - block_size

            blocks.append((i, j))

    k = 0
    for i, j in blocks:
        k += 1
        new_transform = src.window_transform(
            Window(col_off=j, row_off=i, width=block_size, height=block_size)
        )
        src_profile["transform"] = new_transform

        out_img_data = src_data[:, i: i + block_size, j: j + block_size]

        out_img_path = output_img_dir + '/' + output_file_name + '_' + str(k) + '.png' # 制作超分数据集记得修改后缀x2
        with rio.open(out_img_path, "w", **src_profile) as out_src:
            out_src.write(out_img_data)

    src.close()


if __name__ == "__main__":

    input_img_dir = r"F:\project_UAV_GF2_2\4-clip_img_UAV_321_8bit_enhanced-X2"
    output_img_dir = r"F:\project_UAV_GF2_2\5-clip_img_UAV_321_8bit_enhanced-X2_clip"
    block_size = 240
    overlap = 0

    listpic = fnmatch.filter(os.listdir(input_img_dir), '*.tif')

    for img in listpic:
        src_img = input_img_dir + '/' + img
        cut_img(src_img, output_img_dir, img[:-4], block_size, overlap)
