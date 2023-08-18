'''
https://zhuanlan.zhihu.com/p/599978857
'''


import cv2
import tqdm
import numpy as np
import rasterio.mask
import rasterio as rio

from geopandas import GeoDataFrame

def rgb2color(rgb=(0,0,0)):
    r, g, b = rgb
    # 使用opencv内置函数进行BGR2HSV
    bgr_img = np.array([b,g,r],np.uint8).reshape(1,1,3)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = hsv_img[0,0,0]
    s = hsv_img[0,0,1]
    v = hsv_img[0,0,2]

    # 常用颜色hsv值域范围 参考:https://www.cnblogs.com/wangyblzu/p/5710715.html
    # h:[0,180], s:[0,255], v:[0.255]
    color_CNnames = ["黑", "灰", "白","红", "红", "橙", "黄", "绿", "青", "蓝", "紫"]
    color_names = ["black", "gray", "white", "red", "red", "orange", "yellow", "green", "cyan", "blue", "purple"]
    h_mins = [0, 0, 0, 0, 156, 11, 26, 35, 78, 100, 125]
    h_maxs = [180, 180, 180, 10, 180, 25, 34, 77, 99, 124, 155]
    s_mins = [0, 0, 0, 43, 43, 43, 43, 43, 43, 43, 43]
    s_maxs = [255, 43, 30, 255, 255, 255, 255, 255, 255, 255, 255]
    v_mins = [0, 46, 221, 46, 46, 46, 46, 46, 46, 46, 46]
    v_maxs = [46, 220, 255, 255, 255, 255, 255, 255, 255, 255, 255]

    # hsv与值域范围进行匹配,全部匹配不上的话返回unknown
    color_CNname = "未知"
    color_name = "unknown"
    for i in range(len(color_names)):
        h_min, h_max, s_min, s_max, v_min, v_max = h_mins[i], h_maxs[i], s_mins[i], s_maxs[i], v_mins[i], v_maxs[i]
        if h_min<=h<=h_max and s_min<=s<=s_max and v_min<=v<=v_max:
            color_CNname = color_CNnames[i]
            color_name = color_names[i]
            break
    return color_CNname, color_name
    
    

# 读入矢量和栅格文件
shp_path = "矢量文件名.shp"
raster_path = "栅格文件名.tif"
out_path = "添加颜色字段的新矢量文件名.shp"

shp_data = GeoDataFrame.from_file(shp_path)
raster_data = rio.open(raster_path)

out_shp_data = shp_data.copy()
# 投影变换，使矢量数据与栅格数据投影参数一致
shp_data = shp_data.to_crs(raster_data.crs)

color_CNname_field = []
color_name_field = []
for i in tqdm.tqdm(range(0, len(shp_data))):
    # 获取第i个要素的geometry,并转为GeoJSON格式
    geometry = shp_data.geometry[i]
    geometry_json = [geo.__geo_interface__]
    # 通过geometry_json裁剪栅格影像
    out_image, out_transform = rio.mask.mask(raster_data, geometry_json, all_touched=True, crop=True, nodata=0)
    # 只保留BGR
    out_image = out_image[:3]
    # 转为float32以便将nodata设置为np.nan
    out_image = out_image.astype(np.float32)
    out_image[out_image==0] = np.nan
    # 计算BGR的均值
    mean_b = int(np.nanmean(out_image[0]))
    mean_g = int(np.nanmean(out_image[1]))
    mean_r = int(np.nanmean(out_image[2]))
    mean_rgb = (mean_r,mean_g,mean_b)
    color_CNname, color_name = rgb2color(mean_rgb)
    color_CNname_field.append(color_CNname)
    color_name_field.append(color_name)
    # print(mean_rgb, color_CNname, color_name)
    # break

# 增加属性字段
out_shp_data.insert(out_shp_data.shape[1], 'colorCN', color_CNname_field)
out_shp_data.insert(out_shp_data.shape[1], 'color', color_name_field)
# 导出为shp文件, utf8编码支持中文
out_shp_data.to_file(out_path, encoding='utf8')