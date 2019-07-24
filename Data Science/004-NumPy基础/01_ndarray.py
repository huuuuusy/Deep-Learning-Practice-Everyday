"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： NumPy操作，参考《利用Python进行数据分析》4.1
"""

import numpy as np

"""
生成ndarry
"""
# 从list生成ndarray
print('Example 1:')
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
print(arr1.dtype)

# 长度相同的List将生成多维ndarray
print('Example 2:')
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim)
print(arr2.shape)
print(arr2.dtype)

# 创建
print('Example 3:')
print(np.zeros(10))
print(np.ones((3,6)))
print(np.empty((2,3,2))) # 创建没有初始化值的数组

# arange()函数生成数组，类似于range
print('Example 4:')
print(np.arange(15))

print('Example 5:')

print('Example 6:')

print('Example 7:')

print('Example 8:')

print('Example 9:')

print('Example 10:')

print('Example 11:')