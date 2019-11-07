"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
      PyTorch == 1.2.0
介绍： 参考《深度学习框架PyTorch入门与实践》第二章,介绍神经网络
"""

"""
Autograd实现了反向传播功能，但是直接用来写深度学习的代码在很多情况下还是稍显复杂，torch.nn是专门为神经网络设计的模块化接口。、
nn构建于 Autograd之上，可用来定义和运行神经网络
nn.Module是nn中最重要的类，可把它看成是一个网络的封装，包含网络各层定义以及forward方法，调用forward(input)方法，可返回前向传播的结果
以最早的卷积神经网络：LeNet为例，查看如何用nn.Module实现
LeNet是一个基础的前向传播(feed-forward)网络: 接收输入，经过层层传递运算，得到输出
"""

import torch as t
print(t.__version__)

"""
定义网络
定义网络时，需要继承nn.Module，并实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中
如果某一层(如ReLU)不具有可学习的参数，则既可以放在构造函数中，也可以不放，但建议不放在其中，而在forward中使用nn.functional代替
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
      def __init__(self):
            # nn.Module子类的函数必须在构造函数中执行父类的构造函数
            super(Net, self).__init__() # 等价于nn.Module.__init__(self)
            
            self.conv1 = nn.Conv2d(1,6,5) # 卷积层，'1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120) # 仿射层/全连接层，y = Wx + b
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84,10)

      def forward(self, x):
            # 卷积-->激活-->池化
            x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
            x = F.max_pool2d(F.relu(self.conv2(x)),2)
            # reshape,-1表示尺度自适应
            x = x.view(x.size()[0],-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

print('================================')

"""
只要在nn.Module的子类中定义了forward函数，backward函数就会自动被实现(利用autograd)
在forward 函数中可使用任何tensor支持的函数，还可以使用if、for循环、print、log等Python语法，写法和标准的Python写法一致
网络的可学习参数通过net.parameters()返回，net.named_parameters可同时返回可学习的参数及名称
forward函数的输入和输出都是Tensor。
"""
params = list(net.parameters())
print(len(params)) # 10,表明网络中有10个可学习参数
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())
# conv1.weight : torch.Size([6, 1, 5, 5])
# conv1.bias : torch.Size([6])
# conv2.weight : torch.Size([16, 6, 5, 5])
# conv2.bias : torch.Size([16])
# fc1.weight : torch.Size([120, 400])
# fc1.bias : torch.Size([120])
# fc2.weight : torch.Size([84, 120])
# fc2.bias : torch.Size([84])
# fc3.weight : torch.Size([10, 84])
# fc3.bias : torch.Size([10])

input = t.randn(1, 1, 32, 32)
out = net(input)
print(out.size()) 
# torch.Size([1, 10])
net.zero_grad() # 所有参数的梯度清零
out.backward(t.ones(1,10)) # 反向传播

print('================================')

"""
损失函数
nn实现了神经网络中大多数的损失函数，例如nn.MSELoss用来计算均方误差，nn.CrossEntropyLoss用来计算交叉熵损失
"""
output = net(input)
target = t.arange(0,10).view(1,10) 
target = target.float() # 需要将target转化为float，否则会报错
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss) # loss是个scalar
# tensor(28.5457, grad_fn=<MseLossBackward>)

# 运行.backward，观察调用之前和调用之后的grad
net.zero_grad() # 把net中所有可学习参数的梯度清零
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad) # tensor([0., 0., 0., 0., 0., 0.])
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad) # tensor([ 0.0324,  0.0986, -0.1076,  0.0462,  0.0325,  0.0636])

print('================================')

"""
优化器
在反向传播计算完所有参数的梯度后，还需要使用优化方法来更新网络的权重和参数，例如随机梯度下降法(SGD)的更新策略如下：
weight = weight - learning_rate * gradient
手动实现如下：
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)# inplace 减法
torch.optim中实现了深度学习中绝大多数的优化方法，例如RMSProp、Adam、SGD等，更便于使用，因此大多数时候并不需要手动写上述代码
"""

import torch.optim as optim
#新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# 在训练过程中
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad() 

# 计算损失
output = net(input)
loss = criterion(output, target)

#反向传播
loss.backward()

#更新参数
optimizer.step()
print(optimizer)
# SGD (
# Parameter Group 0
#     dampening: 0
#     lr: 0.01
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )