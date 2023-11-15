#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
均匀滑窗裁剪
~~~~~~~~~~~~~~~~
code by ZC
Aerospace Information Research Institute, Chinese Academy of Sciences
"""


from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
from random import random
import os


def cut_img(
    src_img: Path,
    label_img: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    output_file_name: str,
    block_size: int,
    overlap: float,
):
    overlap = int(overlap * block_size)
    src = rio.open(src_img)
    label = rio.open(label_img)
    src_data = src.read()
    label_data = label.read()

    src_profile = src.profile
    src_profile.update(
        {
            "height": block_size,
            "width": block_size,
            "compress": "deflate",
        }
    )
    label_profile = label.profile
    label_profile.update(
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
            if height - i < block_size:
                i = height - block_size
            if width - j < block_size:
                j = width - block_size

            blocks.append((i, j))

    max_value = 255 if np.max(label_data) == 1 else 1
    area = block_size * block_size
    img_threshold = area * 0.5
    label_threshold = area * 0.01
    k = 0
    for i, j in blocks:
        k += 1
        new_transform = src.window_transform(
            Window(col_off=j, row_off=i, width=block_size, height=block_size)
        )
        src_profile["transform"] = new_transform
        label_profile["transform"] = new_transform

        out_img_data = src_data[:, i : i + block_size, j : j + block_size]
        out_label_data = label_data[:, i : i + block_size, j : j + block_size]

        img_non_zero = np.count_nonzero(np.count_nonzero(out_img_data, axis=0))
        label_non_zero = np.count_nonzero(out_label_data)

        select = img_non_zero > img_threshold and label_non_zero > label_threshold

        if select or (
            np.all(label_non_zero == 0)
            and img_non_zero > img_threshold
            and random() < 0.1
        ):
            out_img_path = output_img_dir / f"{output_file_name}_{k}.tif"
            out_label_path = output_label_dir / f"{output_file_name}_{k}.tif"
            with rio.open(out_img_path, "w", **src_profile) as out_src:
                out_src.write(out_img_data)
            with rio.open(out_label_path, "w", **label_profile) as out_label_path:
                out_label_path.write(out_label_data * max_value)

    src.close()
    label.close()


def main(
    input_img_dir,
    input_label_dir,
    output_img_dir,
    output_label_dir,
    block_size,
    overlap,
    num_workers,
):
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # 创建一个包含n个进程的进程池
    pool = Pool(processes=num_workers)
    # 准备一些参数
    arguments = [
        (
            i,
            input_label_dir / i.name,
            output_img_dir,
            output_label_dir,
            i.stem,
            block_size,
            overlap,
        )
        for i in input_img_dir.iterdir()
    ]

    # 使用 map 方法批量运行函数
    pool.starmap(cut_img, arguments)
    # 关闭进程池，不再接受新的任务
    pool.close()
    # 等待所有进程完成
    pool.join()


if __name__ == "__main__":

    os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

    input_img_dir = Path(r"E:\project_daijiandi\0-other_data\image_tif")
    input_label_dir = Path(r"E:\project_daijiandi\0-other_data\label_tif")
    output_img_dir = Path(r"E:\project_daijiandi\1-clip_img")
    output_label_dir = Path(r"E:\project_daijiandi\1-raster_label")
    block_size = 1024
    overlap = 0
    num_workers = 6

    main(
        input_img_dir,
        input_label_dir,
        output_img_dir,
        output_label_dir,
        block_size,
        overlap,
        num_workers,
    )
