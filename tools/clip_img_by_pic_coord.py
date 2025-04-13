import os
from osgeo import gdal, osr

def crop_image_by_pixel(input_path, output_path, x_offset, y_offset, width, height):
    """
    从指定像素位置开始裁剪影像
    
    参数:
        input_path: 输入影像路径
        output_path: 输出影像路径
        x_offset: 左上角X像素坐标(从0开始)
        y_offset: 左上角Y像素坐标(从0开始)
        width: 要裁剪的宽度(像素)
        height: 要裁剪的高度(像素)
    """
    # 打开原始影像
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        raise ValueError("无法打开输入影像文件")
    
    # 确保裁剪范围在图像范围内
    if (x_offset < 0 or y_offset < 0 or 
        x_offset + width > src_ds.RasterXSize or 
        y_offset + height > src_ds.RasterYSize):
        raise ValueError("裁剪范围超出图像边界")
    
    # 获取原始影像信息
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    band_count = src_ds.RasterCount
    data_type = src_ds.GetRasterBand(1).DataType
    
    # 计算裁剪后的新地理变换参数
    new_geotransform = (
        geotransform[0] + x_offset * geotransform[1] + y_offset * geotransform[2],
        geotransform[1],
        geotransform[2],
        geotransform[3] + x_offset * geotransform[4] + y_offset * geotransform[5],
        geotransform[4],
        geotransform[5]
    )
    
    # 创建输出影像
    driver = gdal.GetDriverByName('GTiff')  # 默认使用GeoTIFF格式
    out_ds = driver.Create(output_path, width, height, band_count, data_type)
    out_ds.SetGeoTransform(new_geotransform)
    out_ds.SetProjection(projection)
    
    # 逐个波段读取和写入数据
    for band_num in range(1, band_count + 1):
        src_band = src_ds.GetRasterBand(band_num)
        data = src_band.ReadAsArray(x_offset, y_offset, width, height)
        
        out_band = out_ds.GetRasterBand(band_num)
        out_band.WriteArray(data)
        
        # 复制波段统计信息和NoData值
        if src_band.GetNoDataValue() is not None:
            out_band.SetNoDataValue(src_band.GetNoDataValue())
        
        # 复制颜色表和描述信息
        if src_band.GetColorTable() is not None:
            out_band.SetColorTable(src_band.GetColorTable())
        out_band.SetDescription(src_band.GetDescription())
    
    # 关闭数据集
    out_ds = None
    src_ds = None
    
    print(f"裁剪完成，结果已保存到: {output_path}")

# 示例用法
if __name__ == "__main__":
    input_image = r"G:\project_UAV_GF2_2\3-clip_img_GF2_432_enhanced\0.tif"  # 输入影像路径
    output_image = r"D:\OneDrive\paper\SCI-6-populus_GF2_UAV\0-pic\流程图中的区域\0x2_未匹配.png"  # 输出影像路径
    x_offset = 365  # 左上角X像素坐标(从0开始)
    y_offset = 170  # 左上角Y像素坐标(从0开始)
    crop_width = 75  # 裁剪宽度(像素)
    crop_height = 75  # 裁剪高度(像素)
    
    crop_image_by_pixel(input_image, output_image, x_offset, y_offset, crop_width, crop_height)