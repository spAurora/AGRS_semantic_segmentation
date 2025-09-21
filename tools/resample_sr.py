# -*- coding: utf-8 -*-

"""
批量修改图片大小
用于超分重建数据集生成
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
Ghent University
Haoyu.Wang@ugent.be
"""
import os
import rasterio
from rasterio.enums import Resampling

def resample_to_match(input_dir_a, reference_dir_b, output_dir_c, resample_factor):
    if not os.path.exists(output_dir_c):
        os.makedirs(output_dir_c)

    for filename in os.listdir(input_dir_a):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_dir_a, filename)
            reference_path = os.path.join(reference_dir_b, filename)
            output_path = os.path.join(output_dir_c, filename)

            if os.path.exists(reference_path):
                # 读取A路径下的图像
                with rasterio.open(input_path) as src:
                    # 读取B路径下的参考图像
                    with rasterio.open(reference_path) as ref_src:
                        # 获取B图像的尺寸
                        ref_width = ref_src.width*resample_factor
                        ref_height = ref_src.height*resample_factor

                        # 重采样A图像以匹配B图像的尺寸
                        data = src.read(
                            out_shape=(
                                src.count,  # 波段数量
                                ref_height,  # 参考高度
                                ref_width    # 参考宽度
                            ),
                            resampling=Resampling.bilinear  # 使用双线性插值
                        )

                        # 更新元数据
                        transform = src.transform * src.transform.scale(
                            (src.width / data.shape[-1]),
                            (src.height / data.shape[-2])
                        )

                        new_meta = src.meta.copy()
                        new_meta.update({
                            'height': ref_height,
                            'width': ref_width,
                            'transform': transform
                        })

                        # 保存到C路径
                        with rasterio.open(output_path, 'w', **new_meta) as dst:
                            dst.write(data)
            else:
                print(f"参考图像 {reference_path} 不存在，跳过 {filename}.")

input_dir_a = r"G:\project_UAV_GF2_2\3-clip_img_UAV_321_8bit_enhanced"  # A路径下的图像
reference_dir_b = r"G:\project_UAV_GF2_2\3-clip_img_GF2_432_enhanced"  # B路径下的参考图像
output_dir_c = r"G:\project_UAV_GF2_2\4-clip_img_UAV_321_8bit_enhanced-X8"  # C路径为输出路径
resample_factor = 8

os.makedirs(output_dir_c, exist_ok=True)

resample_to_match(input_dir_a, reference_dir_b, output_dir_c, resample_factor=resample_factor)