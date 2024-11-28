#!/.conda/envs/dp python
# -*- coding: utf-8 -*-

"""
图片色彩偏移（亮度调整、波段偏移）
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""

from osgeo import gdal
import numpy as np

def increase_brightness_by_band(image_path, output_path, factor=1.2, bias = []):
    # 打开影像文件
    dataset = gdal.Open(image_path)
    
    if not dataset:
        print("无法打开文件")
        return
    
    # 获取图像的波段数量
    band_count = dataset.RasterCount
    
    # 创建一个输出的GDAL文件
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=LZW']  # 使用LZW无损压缩
    output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, band_count, gdal.GDT_Byte, options)
    
    # 循环处理每个波段
    for i in range(band_count):
        # 获取第 i 个波段
        band = dataset.GetRasterBand(i + 1)
        # 读取波段数据到数组
        band_data = band.ReadAsArray()

        # 调整
        band_data = np.clip(band_data*factor+bias[i], 0, 255).astype(np.uint8)
        
        # 写入输出图像
        output_band = output_dataset.GetRasterBand(i + 1)
        output_band.WriteArray(band_data)
        
    # 设置投影和地理变换
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())

    # 关闭数据集
    output_dataset.FlushCache()
    print(f"增强后的图像已保存到: {output_path}")


image_path = r'C:\Users\Administrator\Desktop\暗.jpg'
output_path = r'C:\Users\Administrator\Desktop\暗.jpg'
factor = 0.6
bias = [40, 0, 0, 0]
# 使用函数
output_path = output_path[:-4] + '_' + str(factor) + '.png'
increase_brightness_by_band(image_path, output_path, factor=factor, bias=bias)