"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

MNIST是深度学习领域最常用的数据集，其中包含60000张训练图像和
10000张测试图像。数据集是针对手写数字的分类，每张图是28×28的
灰度图
本例使用最简单的全连接神经网络对MNIST进行分类
"""

import keras
keras.__version__

# 下载数据
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('train image shape:', x_train.shape) # (60000, 28, 28)
print('train label shape:', y_train.shape) # (60000,)
print('test image shape:', x_test.shape) # (10000, 28, 28)
print('test label shape:', y_test.shape) # (10000,)

# 数据准备:将数据归一化到0-1的范围内
x_train = x_train.reshape((60000, 28*28))
x_train = x_train.astype('float32')/255
#print(x_train.shape)
#print(x_train[0])

x_test = x_test.reshape((10000, 28*28))
x_test = x_test.astype('float32')/255

# 准备标签
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y_train.shape)
#print(y_train[0])

# 网络结构
from keras import layers
from keras import models

simple_dnn = models.Sequential()
simple_dnn.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
simple_dnn.add(layers.Dense(10, activation = 'softmax'))

# 编译
simple_dnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 训练
simple_dnn.fit(x_train, y_train, epochs=5, batch_size=128)

# 测试
test_loss, test_acc = simple_dnn.evaluate(x_test, y_test)
print('test loss:', test_loss)
print('test accuracy:', test_acc)