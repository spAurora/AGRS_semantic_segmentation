import os
import numpy as np
from osgeo import gdal
import numpy.ma as ma

def read_tif_file(tif_path):
    """读取TIFF文件并返回数据和地理信息"""
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise ValueError(f"无法打开TIFF文件: {tif_path}")
    
    # 读取数据
    data = dataset.ReadAsArray()
    
    # 获取地理信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    dataset = None
    return data, geotransform, projection

def read_npz_file(npz_path):
    """读取NPZ文件"""
    try:
        npz_data = np.load(npz_path)
        # 假设NPZ文件中只有一个数组，或者使用第一个数组
        if len(npz_data.files) > 0:
            first_key = npz_data.files[0]
            mask_data = npz_data[first_key]
            return mask_data
        else:
            raise ValueError(f"NPZ文件为空: {npz_path}")
    except Exception as e:
        raise ValueError(f"无法读取NPZ文件 {npz_path}: {e}")

def save_tif_file(output_path, data, geotransform, projection, dtype=gdal.GDT_Float32):
    """保存为TIFF文件"""
    driver = gdal.GetDriverByName('GTiff')
    
    if len(data.shape) == 2:
        bands = 1
        height, width = data.shape
    else:
        bands, height, width = data.shape
    
    # 创建输出文件
    out_dataset = driver.Create(output_path, width, height, bands, dtype)
    
    if out_dataset is None:
        raise ValueError(f"无法创建输出文件: {output_path}")
    
    # 设置地理信息
    out_dataset.SetGeoTransform(geotransform)
    out_dataset.SetProjection(projection)
    
    # 写入数据
    if bands == 1:
        out_dataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(bands):
            out_dataset.GetRasterBand(i+1).WriteArray(data[i])
    
    out_dataset = None

def process_prediction_with_mask(pred_folder, mask_folder, output_folder):
    """处理预测结果和掩膜的点乘操作"""
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取预测文件夹中的所有tif文件
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.tif')]
    
    if not pred_files:
        print("在预测文件夹中未找到任何TIFF文件")
        return
    
    processed_count = 0
    
    for pred_file in pred_files:
        try:
            # 构建文件路径
            pred_path = os.path.join(pred_folder, pred_file)
            
            # 构建对应的掩膜文件路径（假设文件名相同，只是扩展名不同）
            mask_file = os.path.splitext(pred_file)[0] + '.npz'
            mask_path = os.path.join(mask_folder, mask_file)
            
            # 检查掩膜文件是否存在
            if not os.path.exists(mask_path):
                print(f"警告: 未找到对应的掩膜文件 {mask_path}，跳过处理 {pred_file}")
                continue
            
            print(f"正在处理: {pred_file}")
            
            # 读取预测结果
            pred_data, geotransform, projection = read_tif_file(pred_path)
            
            # 读取掩膜数据
            mask_data = read_npz_file(mask_path)
            
            # 检查尺寸是否匹配
            if pred_data.shape != mask_data.shape:
                print(f"警告: {pred_file} 和 {mask_file} 尺寸不匹配")
                print(f"预测结果尺寸: {pred_data.shape}, 掩膜尺寸: {mask_data.shape}")
                # 尝试调整掩膜尺寸以匹配预测结果
                if len(pred_data.shape) == len(mask_data.shape):
                    try:
                        mask_data = np.resize(mask_data, pred_data.shape)
                        print("已调整掩膜尺寸以匹配预测结果")
                    except:
                        print("无法调整掩膜尺寸，跳过此文件")
                        continue
                else:
                    print("维度不匹配，跳过此文件")
                    continue
            
            # 执行点乘操作
            result_data = pred_data * mask_data
            
            # 构建输出文件路径
            output_path = os.path.join(output_folder, pred_file)
            
            # 确定输出数据类型
            if pred_data.dtype == np.uint8:
                dtype = gdal.GDT_Byte
            elif pred_data.dtype == np.uint16:
                dtype = gdal.GDT_UInt16
            elif pred_data.dtype == np.int16:
                dtype = gdal.GDT_Int16
            elif pred_data.dtype == np.uint32:
                dtype = gdal.GDT_UInt32
            elif pred_data.dtype == np.int32:
                dtype = gdal.GDT_Int32
            elif pred_data.dtype == np.float32:
                dtype = gdal.GDT_Float32
            elif pred_data.dtype == np.float64:
                dtype = gdal.GDT_Float64
            else:
                dtype = gdal.GDT_Float32  # 默认使用Float32
            
            # 保存结果
            save_tif_file(output_path, result_data, geotransform, projection, dtype)
            
            print(f"已完成: {pred_file}")
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {pred_file} 时出错: {e}")
            continue
    
    print(f"\n处理完成！成功处理 {processed_count} 个文件")
    print(f"结果保存在: {output_folder}")

def main():
    # 设置文件夹路径
    pred_folder = "A"  # 预测结果文件夹
    mask_folder = "B"   # 掩膜文件夹
    output_folder = "result"  # 输出文件夹
    
    # 检查输入文件夹是否存在
    if not os.path.exists(pred_folder):
        print(f"错误: 预测文件夹 {pred_folder} 不存在")
        return
    
    if not os.path.exists(mask_folder):
        print(f"错误: 掩膜文件夹 {mask_folder} 不存在")
        return
    
    # 执行处理
    process_prediction_with_mask(pred_folder, mask_folder, output_folder)

if __name__ == "__main__":
    main()