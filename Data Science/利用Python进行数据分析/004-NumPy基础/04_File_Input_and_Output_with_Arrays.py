"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用数组进行文件输入和输出，参考《利用Python进行数据分析》4.4
"""
import numpy as np

"""
文件的输入输出　
np.save()保存为.npy文件
np.load()读取指定路径的文件
"""
print('Example 1:')
arr = np.arange(10)
np.save('some_array.npy', arr)
print(np.load('some_array.npy'))

"""
压缩文件的输入输出　
np.savez()保存为.npz文件,可以保存多个数组
np.load()读取指定路径的文件
"""
print('Example 1:')
np.savez('array_archive.npz', a=arr, b=arr) # 将多个数组存入一个文件
arch = np.load('array_archive.npz')
print(arch['b'])
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr) # 将多个数组存入一个已经压缩的文件