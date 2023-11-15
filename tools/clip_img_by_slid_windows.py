#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
均匀滑窗裁剪
未测试
~~~~~~~~~~~~~~~~
code by ZC
Aerospace Information Research Institute, Chinese Academy of Sciences
"""

from pathlib import Path
from tqdm import tqdm
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from multiprocessing import Pool
from random import random


def cut_img(
    src_img: Path,
    output_img_dir: Path,
    label_img: Path,
    output_label_dir: Path,
    output_file_name: str,
    block_size: int,
    overlap: float,
):
    overlap = int(overlap * block_size)
    with rio.open(src_img) as src, rio.open(label_img) as label:
        src_data = src.read()
        label_data = label.read()

        height, width = src.shape
        blocks = []
        for i in range(0, height, block_size - overlap):
            for j in range(0, width, block_size - overlap):
                if height - i < block_size:
                    i = height - block_size
                if width - j < block_size:
                    j = width - block_size

                blocks.append((i, j))

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

        area = block_size * block_size
        img_threshold = area * 0.5
        label_threshold = area * 0.01

        for i, j in blocks:
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
                out_img_path = output_img_dir / f"{output_file_name}_{i}_{j}.tif"
                out_label_path = output_label_dir / f"{output_file_name}_{i}_{j}.tif"
                with (
                    rio.open(out_img_path, "w", **src_profile) as out_src,
                    rio.open(out_label_path, "w", **label_profile) as out_label_path,
                ):
                    out_src.write(out_img_data)
                    out_label_path.write(out_label_data * 255)


def exec_province(
    province_path, output_img_path, output_label_path, block_size, overlap
):
    province_index = province_path.name[0:2]

    for region_path in province_path.iterdir():
        region_code = region_path.name[0:7]

        src_img = region_path / "01_img_png" / f"{region_code}.tif"
        label_img = region_path / "02_label_png" / f"{region_code}.tif"

        if src_img.exists() and label_img.exists():
            output_file_name = f"{province_index}_{region_code}"
            cut_img(
                src_img,
                output_img_path,
                label_img,
                output_label_path,
                output_file_name,
                block_size,
                overlap,
            )


def main(
    input_path: Path,
    output_img_path: Path,
    output_label_path: Path,
    block_size,
    overlap,
    num_workers,
):
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)

    # 创建一个包含4个进程的进程池
    pool = Pool(processes=num_workers)
    # 准备一些参数
    arguments = [
        (i, output_img_path, output_label_path, block_size, overlap)
        for i in input_path.iterdir()
    ]
    # 使用 map 方法批量运行函数
    pool.starmap(exec_province, arguments)
    # 关闭进程池，不再接受新的任务
    pool.close()
    # 等待所有进程完成
    pool.join()


if __name__ == "__main__":
    input_path = Path(r"D:\东北\原始")
    output_img_path = Path(r"D:\东北\clip_img")
    output_label_path = Path(r"D:\东北\raster_label")
    block_size = 1024
    overlap = 0
    num_workers = 8

    main(
        input_path, output_img_path, output_label_path, block_size, overlap, num_workers
    )
