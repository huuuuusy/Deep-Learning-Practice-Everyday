"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： np.random系列随机数生成函数学习
参考： https://www.cnblogs.com/lemonbit/p/6864179.html
"""

import numpy as np

"""
numpy.random.rand(d0,d1,…,dn)
    rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
    返回值为指定维度的ndarray
"""
print(np.random.rand(4,2))
#[[0.39561594 0.62189809]
# [0.63264199 0.79004789]
# [0.78957062 0.9487078 ]
# [0.22266734 0.46988012]]

"""
numpy.random.randn(d0,d1,…,dn)
    randn函数根据给定维度生成满足标准正态分布的样本
    返回值为指定维度的ndarray
"""
print(np.random.randn(4,2))
#[[ 2.10366997e-01  9.54164156e-02]
# [-5.56298587e-01 -8.94611643e-01]
# [-6.54609480e-04  8.20979548e-01]
# [ 5.32734232e-01 -2.79085202e-01]]

"""
numpy.random.randint(low, high=None, size=None, dtype=’l’)
    返回随机整数，范围区间为[low,high），包含low，不包含high
    参数：low为最小值，high为最大值，size为数组维度大小，dtype为数据类型，默认的数据类型是np.int
    high没有填写时，默认生成随机数的范围是[0，low)
"""
print(np.random.randint(1,size=5)) # 返回[0,1)之间的整数，所以只有0
# [0 0 0 0 0]
print(np.random.randint(1,5)) # 返回1个[1,5)时间的随机整数
# 2 
print(np.random.randint(-5,5,size=(2,2))) # 返回[-5, 5)范围内的大小为2*2的整数数组
#[[-4  3]
# [ 0  4]]