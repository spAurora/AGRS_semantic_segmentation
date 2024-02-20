import numpy as np
from PIL import Image

def modify_image_colors(image_path, r_offset, g_offset, b_offset):
    # 打开图像
    image = Image.open(image_path)

    # 转换图像为NumPy数组
    img_array = np.array(image, dtype=np.float32)

    # 修改颜色通道
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + r_offset, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + g_offset, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] + b_offset, 0, 255)

    # 转换回PIL图像
    modified_image = Image.fromarray(img_array.astype(np.uint8))

    # 保存修改后的图像
    modified_image_path = r"E:\xinjiang_huyang_hongliu\paper_240114补充\3-清晰样本-ASMLV3\0-修改颜色\output.tif"
    modified_image.save(modified_image_path)

    print(f"图像颜色通道已修改，保存为 {modified_image_path}")

# 使用示例：对图像进行 R-8，G+8, B-8 的修改
image_path = r"E:\xinjiang_huyang_hongliu\paper_240114补充\3-清晰样本-ASMLV3\0-修改颜色\1705969098394.jpg"
modify_image_colors(image_path, 10, 10, -12)