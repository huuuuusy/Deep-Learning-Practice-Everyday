"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
      PyTorch == 1.2.0
介绍： 参考《深度学习框架PyTorch入门与实践》第二章,介绍Tensor
"""

"""
Tensor是PyTorch中重要的数据结构，可认为是一个高维数组。
它可以是一个数（标量）、一维数组（向量）、二维数组（矩阵）以及更高维的数组。
Tensor和Numpy的ndarrays类似，但Tensor可以使用GPU进行加速。
Tensor的使用和Numpy及Matlab的接口十分相似。
"""

import torch as t
print(t.__version__)

# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)
x = t.Tensor([[1,2],[3,4]])
print(x)
