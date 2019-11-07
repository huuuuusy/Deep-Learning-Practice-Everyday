"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
      PyTorch == 1.2.0
介绍： 参考《深度学习框架PyTorch入门与实践》第二章,介绍autograd
"""

"""
深度学习的算法本质上是通过反向传播求导数，而PyTorch的autograd模块则实现了此功能
在Tensor上的所有操作，autograd都能为它们自动提供微分，避免了手动计算导数的复杂过程
要想使得Tensor使用autograd功能，只需要设置tensor.requries_grad=True
"""

import torch as t
print(t.__version__)

"""
为tensor设置 requires_grad 标识，代表着需要求导数
pytorch 会自动调用autograd 记录操作
"""
print('Example 1:')
x = t.ones(2, 2, requires_grad=True)
# 上一步等价于
# x = t.ones(2,2)
# x.requires_grad = True
print(x)
# tensor([[1., 1.],
#        [1., 1.]], requires_grad=True)

y = x.sum()
print(y) 
# tensor(4., grad_fn=<SumBackward0>)

"""
反向传播求梯度
"""
print('Example 2:')
y.backward() # 反向传播，计算梯度
# y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# 对于x.grad，其中每个值的梯度都为1
print(x.grad)
# tensor([[1., 1.],
#        [1., 1.]])

y.backward()
print(x.grad)
# tensor([[2., 2.],
#        [2., 2.]])

y.backward()
print(x.grad)
# tensor([[3., 3.],
#        [3., 3.]])

"""
grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度
所以反向传播之前需把梯度清零
"""
print('Example 3:')
# 以下划线结束的函数是inplace操作，会修改自身的值，就像add_
x.grad.data.zero_() # 将x.grad置零
y.backward()
print(x.grad)
# tensor([[ 1.,  1.],
#        [ 1.,  1.]])
