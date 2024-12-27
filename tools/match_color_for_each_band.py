import os
import numpy as np
from osgeo import gdal
from skimage import exposure

def read_gdal_image(image_path):
    """ 使用GDAL读取影像数据，返回影像数据和波段数量 """
    dataset = gdal.Open(image_path)
    if not dataset:
        raise Exception(f"无法打开影像文件：{image_path}")
    
    # 获取影像的波段数
    bands = dataset.RasterCount
    # 获取影像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    # 读取每个波段的数据
    image_data = []
    for i in range(1, bands + 1):
        band = dataset.GetRasterBand(i)
        band_data = band.ReadAsArray()
        image_data.append(band_data)
    
    # 将各个波段数据合并成一个数组
    return np.array(image_data), width, height

def save_gdal_image(output_path, image_data, ref_image_path):
    """ 使用GDAL保存影像数据到文件 """
    ref_dataset = gdal.Open(ref_image_path)
    driver = gdal.GetDriverByName('GTiff')
    
    # 创建输出影像文件
    output_dataset = driver.Create(output_path, ref_dataset.RasterXSize, ref_dataset.RasterYSize, 
                                   len(image_data), gdal.GDT_Byte)
    output_dataset.SetGeoTransform(ref_dataset.GetGeoTransform())
    output_dataset.SetProjection(ref_dataset.GetProjection())
    
    # 将每个波段数据写入影像文件
    for i, band_data in enumerate(image_data, start=1):
        output_band = output_dataset.GetRasterBand(i)
        output_band.WriteArray(band_data)
    
    output_dataset.FlushCache()

def process_image_pair(low_res_path, high_res_path, output_path):
    """ 处理低分辨率和高分辨率影像的直方图匹配 """
    # 读取低分辨率和高分辨率影像
    low_res_img, width, height = read_gdal_image(low_res_path)
    high_res_img, _, _ = read_gdal_image(high_res_path)

    # 对每个波段进行直方图匹配
    matched_bands = []
    for low_band, high_band in zip(low_res_img, high_res_img):
        matched_band = exposure.match_histograms(low_band, high_band)
        matched_bands.append(matched_band)

    # 保存处理后的影像
    save_gdal_image(output_path, matched_bands, low_res_path)
    print(f"处理完成：{output_path}")

def batch_process_images(low_res_folder, high_res_folder, output_folder):
    """ 批量处理低分辨率和高分辨率影像 """
    # 获取低分辨率文件夹中的所有影像文件
    low_res_files = os.listdir(low_res_folder)
    
    for file_name in low_res_files:
        # 构造低分辨率和高分辨率影像的文件路径
        low_res_path = os.path.join(low_res_folder, file_name)
        high_res_path = os.path.join(high_res_folder, file_name)

        # 检查高分辨率影像是否存在
        if os.path.exists(high_res_path):
            # 构造输出路径
            output_path = os.path.join(output_folder, file_name)
            # 处理影像对
            process_image_pair(low_res_path, high_res_path, output_path)
        else:
            print(f"高分辨率影像 {high_res_path} 不存在，跳过该文件。")

# 设置文件夹路径
low_res_folder = r'F:\project_UAV_GF2_2\3-clip_img_GF2_432_enhanced'  # 低分辨率影像文件夹路径
high_res_folder = r'F:\project_UAV_GF2_2\4-clip_img_UAV_321_8bit_enhanced-X2'  # 高分辨率影像文件夹路径
output_folder = r'F:\project_UAV_GF2_2\5-clip_img_GF2_432_enhanced_match'  # 输出影像文件夹路径

# 执行批量处理
batch_process_images(low_res_folder, high_res_folder, output_folder)