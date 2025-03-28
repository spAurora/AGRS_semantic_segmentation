import os
import cv2

# 定义文件夹路径
folder_A = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\xinjiang-sentinel2-10m-xzy\images-123'  # 替换为文件夹A的实际路径
folder_B = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\xinjiang-sentinel2-10m-xzy\labels'  # 替换为文件夹B的实际路径
output_folder = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\xinjiang-sentinel2-10m-xzy\images-covered'  # 替换为输出文件夹的实际路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹A中的所有tif文件
tif_files = [f for f in os.listdir(folder_A) if f.lower().endswith('.tif')]

for tif_file in tif_files:
    # 检查文件夹B中是否存在同名文件
    if not os.path.exists(os.path.join(folder_B, tif_file)):
        print(f"警告：文件夹B中不存在文件 {tif_file}，跳过处理")
        continue
    
    try:
        # 读取彩色图像（文件夹A）
        img_color = cv2.imread(os.path.join(folder_A, tif_file), cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError(f"无法读取彩色图像 {tif_file}")
        
        # 读取灰度图像（文件夹B）
        img_gray = cv2.imread(os.path.join(folder_B, tif_file), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"无法读取灰度图像 {tif_file}")
        
        # 检查图像尺寸是否匹配
        if img_color.shape[:2] != img_gray.shape:
            raise ValueError(f"图像 {tif_file} 尺寸不匹配")
        
        # 找到灰度图像中值为255的像素位置
        mask = img_gray == 255
        
        # 将这些位置的像素值改为(255, 255, 0)（黄色）
        img_color[mask] = [0, 255, 255]  # OpenCV使用BGR顺序
        
        # 保存处理后的图像
        output_path = os.path.join(output_folder, tif_file)
        cv2.imwrite(output_path, img_color)
        
        print(f"已处理并保存: {tif_file}")
    
    except Exception as e:
        print(f"处理文件 {tif_file} 时出错: {str(e)}")

print("处理完成！")