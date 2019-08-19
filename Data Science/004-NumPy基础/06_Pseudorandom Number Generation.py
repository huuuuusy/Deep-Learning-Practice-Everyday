"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用NumPy进行伪随机数的生成，参考《利用Python进行数据分析》4.6
"""
import numpy as np
"""
伪随机数生成
np.random中的数据生成函数公用一个全局的随机数生成种子
np.random.seed()用于更改随机数生成器中的随机数种子
np.random.RandomState生成一个随机数生成器，使数据独立于其他的随机数状态
"""
print('Example 1:')
samples = np.random.normal(size=(2, 2)) # np.random.normal()生成指定大小的正态分布样本数组
print(samples) # [[-1.16594047  0.11405387] [ 0.05114134  0.43272191]]

print('Example 2:')
rng = np.random.RandomState(1234)
print(rng.randn(10)) # [ 0.47143516 -1.19097569  1.43270697 -0.3126519  -0.72058873  0.88716294  0.85958841 -0.6365235   0.01569637 -2.24268495]
