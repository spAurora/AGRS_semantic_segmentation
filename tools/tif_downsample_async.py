"""
对一个目录下的所有影像进行下采样, 转移到一个新的目录里
异步加速
~~~~~~~~~~~~~~~~
code by Zhang Chi
Aerospace Information Research Institute, Chinese Academy of Sciences
University of Chinese Academy of Sciences
yiguanxianyu@gmail.com
zhangchi233@mails.ucas.ac.cn
"""
import os
import asyncio

async def process_tif(input_file_path, output_file_path, ratio_percentage):
    """
    异步处理tif文件
    """
    process = await asyncio.create_subprocess_exec(
        "gdal_translate",
        "-outsize",
        ratio_percentage,
        ratio_percentage,
        "-r",
        "bilinear",
        input_file_path,
        output_file_path,
    )
    await process.wait()

async def main():
    ratio = 0.2
    src_folder = "/Users/xianyu/code/downsampling/ToZC"  # 源文件夹路径
    dest_folder = "/Users/xianyu/code/downsampling/output"  # 目标文件夹路径

    ratio_percentage = f"{int(ratio*100)}%"

    tasks = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".tif"):
                input_file_path = os.path.join(root, file)

                # 构建目标路径
                relative_path = os.path.relpath(root, src_folder)
                target_dir = os.path.join(dest_folder, relative_path)
                output_file_path = os.path.join(target_dir, file)

                os.makedirs(target_dir, exist_ok=True)

                # 添加到任务列表
                tasks.append(process_tif(input_file_path, output_file_path, ratio_percentage))

    # 并发执行所有任务
    await asyncio.gather(*tasks)

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
