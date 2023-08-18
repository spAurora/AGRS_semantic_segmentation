'''
作者：王振庆
链接: https://zhuanlan.zhihu.com/p/536344010
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''
from osgeo import gdal, ogr, osr
import os


def raster2vector(raster_path, vecter_path, field_name="class", ignore_values = None):
    
    # 读取路径中的栅格数据
    raster = gdal.Open(raster_path)
    # in_band 为想要转为矢量的波段,一般需要进行转矢量的栅格都是单波段分类结果
    # 若栅格为多波段,需要提前转换为单波段
    band = raster.GetRasterBand(1)
    
    # 读取栅格的投影信息,为后面生成的矢量赋予相同的投影信息
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())
    
    
    drv = ogr.GetDriverByName("ESRI Shapefile")
    # 若文件已经存在,删除
    if os.path.exists(vecter_path):
        drv.DeleteDataSource(vecter_path)
        
    # 创建目标文件
    polygon = drv.CreateDataSource(vecter_path)
    # 创建面图层
    poly_layer = polygon.CreateLayer(vecter_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)
    # 添加浮点型字段,用来存储栅格的像素值
    field = ogr.FieldDefn(field_name, ogr.OFTReal)  
    poly_layer.CreateField(field)
    
    # FPolygonize将每个像元转成一个矩形，然后将相似的像元进行合并
    # 设置矢量图层中保存像元值的字段序号为0
    gdal.FPolygonize(band, None, poly_layer, 0)
    
    # 删除ignore_value链表中的类别要素
    if ignore_values is not None:
        for feature in poly_layer:
            class_value = feature.GetField('class')
            for ignore_value in ignore_values:
                if class_value==ignore_value:
                    # 通过FID删除要素
                    poly_layer.DeleteFeature(feature.GetFID())
                    break
                
    polygon.SyncToDisk()
    polygon = None

 
if __name__ == '__main__':
    
    raster_path = r"demo.tif"
    vecter_path = r"demo.shp"
    field_name = "class"
    # 第0类删除,若实际情况不需要1类和2类,则ignore_values = [1,2]
    ignore_values = [0]
    raster2vector(raster_path, vecter_path, field_name=field_name, ignore_values = ignore_values)