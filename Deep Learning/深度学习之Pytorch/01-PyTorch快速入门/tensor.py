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

"""
Tensor和Numpy的数组之间的互操作非常容易且快速
对于Tensor不支持的操作，可以先转为Numpy数组处理，之后再转回Tensor
"""
print('Example 6:')
a = t.ones(5)# 新建一个全1的Tensor
print(a) # tensor([1., 1., 1., 1., 1.])
b = a.numpy() # Tensor -> Numpy
print(b) # [1. 1. 1. 1. 1.]
import numpy as np
a = np.ones(5) 
b = t.from_numpy(a) # Numpy->Tensor
print(a) # [1. 1. 1. 1. 1.]
print(b) # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

"""
Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源
但这也意味着，如果其中一个变了，另外一个也会随之改变
"""
print('Example 7:')
b.add_(1) # 以`_`结尾的函数会修改自身
print(a) # [2. 2. 2. 2. 2.]
print(b) # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
# 因为b是从a中转换来的，所以a和b共享内存，当其中一个改变时，另一个也随之改变

"""
利用scalar.item从tensor中取出数据
直接tensor[idx]得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.
利用scalar.item可以从scalar中取出具体的数值
"""
print('Example 8:')
scalar = b[0]
print(scalar) # tensor(2., dtype=torch.float64)
print(scalar.size()) # torch.Size([]),scalar是一个0-dim的tensor
print(scalar.item()) # 2.0,利用scalar.item取出具体的数值

print('Example 9:')
tensor = t.tensor([2]) # 注意和scalar的区别
print(tensor) # tensor([2])，这里是一个1-dim的tensor
print(scalar) # tensor(2., dtype=torch.float64),这里是一个0-dim的tensor

print('Example 10:')
print(tensor.size()) # torch.Size([1])
print(scalar.size()) # torch.Size([])

# 只有一个元素的tensor也可以调用`tensor.item()`
print('Example 11:')
print(tensor.item()) # 2
print(scalar.item()) # 2.0

"""
在pytorch中还有一个和np.array 很类似的接口: torch.tensor
"""
print('Example 12:')
tensor = t.tensor([3,4]) # 新建一个包含 3，4 两个元素的tensor
scalar = t.tensor(3)
print(scalar) # tensor(3)

print('Example 13:')
old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
print(old_tensor) # tensor([3, 4])
print(new_tensor) # tensor([1111,    4])

"""
t.tensor()总是会进行数据拷贝，新tensor和原来的数据不再共享内存
如果想共享内存，建议使用torch.from_numpy()或者tensor.detach()来新建一个tensor, 二者共享内存
"""
new_tensor = old_tensor.detach() # tensor.detach()会建立共享内存的新tensor
new_tensor[0] = 1111
print(old_tensor) # tensor([1111,    4])
print(new_tensor) # tensor([1111,    4])

"""
Tensor可通过.cuda 方法转为GPU的Tensor
"""
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
z = x+y
print(z)
#tensor([[1.2858, 0.8321, 1.6186],
#        [0.2912, 0.6874, 1.4138],
#        [1.3367, 1.3960, 2.8293],
#        [1.3756, 0.3777, 2.6730],
#        [0.3936, 2.3824, 1.0679]], device='cuda:0')