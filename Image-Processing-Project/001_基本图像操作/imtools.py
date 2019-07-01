"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

图像处理常用自定义函数 
"""

"""
01
输入：文件夹路径
输出：图片名称列表
"""

import os

def get_imlist(path):
    imlist = [os.path.join(path, f) for f in os.listdir(path)]
    return imlist


