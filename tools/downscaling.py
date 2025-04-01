import os
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import numpy as np

def downsample_tif(input_path, output_path, scale_factor=2):
    """
    下采样TIF图像并保持坐标系不变
    
    参数:
        input_path: 输入TIF文件路径
        output_path: 输出TIF文件路径
        scale_factor: 下采样比例因子(默认为2)
    """
    with rasterio.open(input_path) as src:
        # 计算新的宽度和高度
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        
        # 读取数据并进行下采样
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average  # 使用平均值进行重采样
        )
        
        # 更新变换矩阵
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
        
        # 写入输出文件
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=new_height,
            width=new_width,
            count=src.count,
            dtype=data.dtype,
            crs=src.crs,
            transform=transform,
            nodata=src.nodata
        ) as dst:
            dst.write(data)

def process_folder(input_folder, output_folder, scale_factor=2):
    """
    处理文件夹中的所有TIF文件
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        scale_factor: 下采样比例因子(默认为2)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"处理文件: {filename}")
            try:
                downsample_tif(input_path, output_path, scale_factor)
                print(f"成功保存: {output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2-new\1-img-enhance-432"  # 替换为你的输入文件夹路径
    output_folder = r"D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2-new\1-img-enhance-432-ds3"  # 替换为你的输出文件夹路径
    
    # 处理文件夹中的所有TIF文件
    process_folder(input_folder, output_folder, scale_factor=3)