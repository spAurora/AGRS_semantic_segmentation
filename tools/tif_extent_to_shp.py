# -*- coding: utf-8 -*-

"""
影像四至范围转shp
~~~~~~~~~~~~~~~~
code by Deepseek-R1
"""
from osgeo import gdal, ogr, osr
import os

def tif_extent_to_shp(input_tif, output_shp):
    """
    将TIFF影像的矩形范围转为相同坐标系的SHP文件
    
    参数：
        input_tif: 输入TIFF路径
        output_shp: 输出SHP路径
    """
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()

    # 1. 打开TIFF文件
    ds = gdal.Open(input_tif)
    if ds is None:
        raise ValueError("无法打开TIFF文件")

    # 2. 获取影像范围和投影
    x_size, y_size = ds.RasterXSize, ds.RasterYSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # 3. 计算矩形四个角点坐标
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + geotransform[1] * x_size
    y_min = y_max + geotransform[5] * y_size

    # 4. 创建SHP文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    out_ds = driver.CreateDataSource(output_shp)
    
    # 5. 设置空间参考
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    
    # 6. 创建面要素
    out_layer = out_ds.CreateLayer("extent", srs=srs, geom_type=ogr.wkbPolygon)
    feature_def = out_layer.GetLayerDefn()
    feature = ogr.Feature(feature_def)
    
    # 7. 构建矩形多边形
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)  # 左下
    ring.AddPoint(x_max, y_min)  # 右下
    ring.AddPoint(x_max, y_max)  # 右上
    ring.AddPoint(x_min, y_max)  # 左上
    ring.AddPoint(x_min, y_min)  # 闭合
    
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    feature.SetGeometry(polygon)
    
    # 8. 保存要素
    out_layer.CreateFeature(feature)
    out_ds = None
    ds = None
    print(f"成功生成矩形范围SHP: {output_shp}")

# 使用示例
if __name__ == "__main__":
    input_tif_path = r'C:\Users\75198\Desktop\高分二号-clip_img_GF2_432_enhanced\0.tif'
    output_extent_shp_path = r'C:\Users\75198\Desktop\高分二号-clip_img_GF2_432_enhanced\extent\0.shp'
    tif_extent_to_shp(input_tif_path, output_extent_shp_path)