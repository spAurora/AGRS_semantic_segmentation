from osgeo import gdal
import os
import sys
import matplotlib.pyplot as plt

def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    del im_dataset

    return im_data, im_proj, im_geotrans

feature_map_file = r'D:\github_respository\Populus_SR_GF2_UAV\results\20250409-152034-gupopulus_250409\vis\lr_feats.tif'
output_dir = r'D:\github_respository\Populus_SR_GF2_UAV\results\20250409-152034-gupopulus_250409\vis\lr_feats'
# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

im_data = read_img(feature_map_file)[0]

height, width = im_data.shape[1], im_data.shape[2]

cmap = plt.get_cmap('RdYlBu')

for i in range(im_data.shape[0]):
    channel_data =im_data[i]

    output_full_path = output_dir + '/' + str(i) + '.png'


    # 创建图形 (尺寸与原始数据相同)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width/100, height/100)  # 转换为英寸单位
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 覆盖整个图形区域
    ax.set_axis_off()  # 关闭坐标轴
    fig.add_axes(ax)

    # 显示特征图 (不进行插值，保持原始像素)
    ax.imshow(channel_data, cmap=cmap, interpolation='none')

    # 保存为无损PNG (不指定DPI，保持原始尺寸)
    plt.savefig(
        output_full_path, 
        bbox_inches='tight',
        pad_inches=0,
        format='png'
    )
    plt.close()