# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
基于到边缘的比例裁剪GIF
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
from PIL import Image

def crop_gif(input_path, output_path, top_percent=20, bottom_percent=10):
    # 打开GIF图像
    with Image.open(input_path) as img:
        # 获取GIF的帧数
        frames = []
        try:
            while True:
                # 复制当前帧
                frame = img.copy()
                frames.append(frame)
                
                # 获取图像尺寸
                width, height = frame.size
                
                # 计算裁剪的尺寸
                top_crop = height * (top_percent / 100)
                bottom_crop = height * (bottom_percent / 100)
                new_height = height - top_crop - bottom_crop
                
                # 裁剪图像
                cropped_frame = frame.crop((0, top_crop, width, height - bottom_crop))
                
                # 保存裁剪后的帧
                frames[-1] = cropped_frame
                
                # 移动到下一帧
                img.seek(img.tell() + 1)
        except EOFError:
            # 处理完所有帧
            pass
        
        # 保存裁剪后的GIF
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=frame.info['duration'], loop=0)

# 使用示例
input_gif_path = r'C:\Users\ASUS\Desktop\20241115塔河中游大图\塔河中游.gif'  # 替换为你的GIF图片路径
output_gif_path = r'C:\Users\ASUS\Desktop\20241115塔河中游大图\塔河中游_clip-10.gif'  # 替换为你想要保存的路径
crop_gif(input_gif_path, output_gif_path)