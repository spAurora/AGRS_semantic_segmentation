import os
import shutil
from pathlib import Path

def copy_all_files(src_dir, dst_dir):
    """
    遍历 src_dir 及其子文件夹下的所有文件，并将其拷贝到 dst_dir 中。
    
    :param src_dir: 源文件夹路径
    :param dst_dir: 目标文件夹路径
    """
    # 确保目标文件夹存在
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # 源文件路径
            src_file_path = os.path.join(root, file)
            # 目标文件路径
            dst_file_path = os.path.join(dst_dir, file)

            # 处理文件名冲突（如果目标文件夹中已存在同名文件）
            counter = 1
            while os.path.exists(dst_file_path):
                name, ext = os.path.splitext(file)
                dst_file_path = os.path.join(dst_dir, f"{name}_{counter}{ext}")
                counter += 1

            # 拷贝文件
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Copied: {src_file_path} -> {dst_file_path}")

    print("All files copied successfully!")

# 使用示例
src_directory = "/path/to/source/folder"  # 替换为你的源文件夹路径
dst_directory = "/path/to/destination/folder"  # 替换为你的目标文件夹路径

copy_all_files(src_directory, dst_directory)