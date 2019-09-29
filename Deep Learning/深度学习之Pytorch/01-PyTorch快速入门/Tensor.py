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

"""
初始化
"""
print('Example 1:')
# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)
x = t.Tensor([[1,2],[3,4]])
print(x)
#tensor([[1., 2.],
#        [3., 4.]])

"""
查看Tensor形状
"""
print('Example 2:')
# 使用[0,1]均匀分布随机初始化二维数组
x = t.rand(5, 3)  
print(x)
#tensor([[0.2667, 0.0354, 0.1632],
#        [0.8978, 0.2583, 0.9399],
#        [0.0253, 0.5161, 0.7489],
#        [0.6370, 0.9629, 0.0921],
#        [0.5484, 0.0473, 0.0621]])
print(x.size()) # 查看x的形状
#torch.Size([5, 3])

"""
Tensor加法
"""
print('Example 3:')
y = t.rand(5, 3)
# 加法的第一种写法
print(x + y)
# 加法的第二种写法
print(t.add(x, y)) 
# 加法的第三种写法：指定加法结果的输出目标为result
result = t.Tensor(5, 3) # 预先分配空间
t.add(x, y, out=result) # 输入到result
print(result)
#tensor([[0.5741, 0.4217, 0.7766],
#        [0.9973, 0.6055, 1.1535],
#        [0.6078, 0.5949, 1.4927],
#        [0.6471, 1.6649, 0.5669],
#        [1.3586, 0.8408, 0.4164]])

"""
普通加法与inplace加法
注意，函数名后面带下划线_ 的函数会修改Tensor本身。
例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。
"""
print('Example 4:')
print('最初y')
print(y)
#tensor([[0.3074, 0.3863, 0.6134],
#        [0.0995, 0.3473, 0.2136],
#        [0.5824, 0.0789, 0.7437],
#       [0.0101, 0.7020, 0.4747],
#        [0.8102, 0.7935, 0.3543]])
print('第一种加法，y的结果')
y.add(x) # 普通加法，不改变y的内容
print(y)
#tensor([[0.3074, 0.3863, 0.6134],
#        [0.0995, 0.3473, 0.2136],
#        [0.5824, 0.0789, 0.7437],
#        [0.0101, 0.7020, 0.4747],
#        [0.8102, 0.7935, 0.3543]])
print('第二种加法，y的结果')
y.add_(x) # inplace 加法，y变了
print(y)
#tensor([[0.5741, 0.4217, 0.7766],
#        [0.9973, 0.6055, 1.1535],
#        [0.6078, 0.5949, 1.4927],
#        [0.6471, 1.6649, 0.5669],
#        [1.3586, 0.8408, 0.4164]])

"""
Tensor的选取
Tensor的选取操作与Numpy类似
"""
print('Example 5:')
print(x[:, 1])
#tensor([0.0354, 0.2583, 0.5161, 0.9629, 0.0473])

print('Example 6:')
print('Example 7:')
print('Example 8:')
print('Example 9:')
print('Example 10:')
print('Example 11:')
