"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
      PyTorch == 1.2.0
介绍： 参考《深度学习框架PyTorch入门与实践》第二章,对CIFAR-10数据集进行分类
"""

import torch as t
print(t.__version__)

import torch.nn as nn
import torch.nn.functional as F

"""
注意，vscode调用pytorch可能会出现问题，需要添加下面两行
参考：https://blog.csdn.net/wangzi371312/article/details/92796320
"""
import multiprocessing
multiprocessing.set_start_method('spawn',True)

"""
CIFAR-10数据集的分类，步骤如下:
    使用torchvision加载并预处理CIFAR-10数据集
    定义网络
    定义损失函数和优化器
    训练网络并更新网络参数
    测试网络
"""

"""
PyTorch提供了一些可极大简化和加快数据处理流程的工具
对于常用的数据集，PyTorch也提供了封装好的接口供用户快速调用，这些数据集主要保存在torchvison中
torchvision实现了常用的图像数据加载功能，例如Imagenet、CIFAR10、MNIST等，以及常用的数据转换操作，这极大地方便了数据加载，并且代码具有可重用性
"""

"""
数据加载及预处理
CIFAR-10是一个常用的彩色图片数据集，它有10个类别: 
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
每张图片都是 3×32×32 ，也即3-通道彩色图片，分辨率为 32×32
"""
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约100M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定

# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]) # 归一化
    
# 训练集
trainset = tv.datasets.CIFAR10(
                    root='/home/hu/tmp/data/', 
                    train=True, 
                    download=True,
                    transform=transform)

trainloader = t.utils.data.DataLoader(
                    trainset, 
                    batch_size=4, # 注意torch必须指定batch_size，即每次处理的数据必须是一个mini-batches，不支持单样本输入
                    shuffle=True, 
                    num_workers=0) # num_workers设置为0是为了避免vscode的raise RuntimeError

# 测试集
testset = tv.datasets.CIFAR10(
                    root='/home/hu/tmp/data/', 
                    train=False, 
                    download=True, 
                    transform=transform)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4, 
                    shuffle=False,
                    num_workers=0) # num_workers设置为0是为了避免vscode的raise RuntimeError

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label]) # ship

# (data + 1) / 2是为了还原被归一化的数据
img = show((data + 1) / 2).resize((100, 100))
img.show()

# Dataloader是一个可迭代的对象，它将dataset返回的每一条数据拼接成一个batch，并提供多线程加速优化和数据打乱等操作
# 当程序对dataset的所有数据遍历完一遍之后，相应的对Dataloader也完成了一次迭代
dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
print(' '.join('%11s'%classes[labels[j]] for j in range(4))) # car        deer        deer        frog
img = show(tv.utils.make_grid((images+1)/2)).resize((400,100))
img.show()

