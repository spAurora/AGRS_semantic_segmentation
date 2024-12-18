"""
转换视频格式
用前执行winget install ffmpeg
安装完后要关闭IDE再启动终端
~~~~~~~~~~~~~~~~
code by Zhang Chi
Aerospace Information Research Institute, Chinese Academy of Sciences
University of Chinese Academy of Sciences
yiguanxianyu@gmail.com
zhangchi233@mails.ucas.ac.cn
"""

from pathlib import Path
import subprocess

for i in Path(r"C:\Users\Administrator\Videos").glob("*.mkv"):
    print(i)
    command = [
        "ffmpeg",
        "-i",
        str(i),
        "-c:v",
        "av1_nvenc",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-cq",
        "30",
        "-preset",
        "slow",
        "-bsf:a",
        "aac_adtstoasc",
        "-s",
        "1920x1080",
        str(i.with_suffix(".cq30.mp4")),
    ]

    subprocess.run(command)
