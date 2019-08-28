"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
框架： TensorFlow-GPU == 1.13.1
介绍： 线性单分类问题分析
"""

"""
肿瘤类别判断问题
    生成肿瘤数据，并判断新的数据是属于良性肿瘤还是恶性肿瘤
    数据的特征包括病人的年龄和肿瘤大小
    标签为良性肿瘤或者恶性肿瘤
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap
from sklearn.preprocessing import OneHotEncoder

"""
生成样本数据集
"""
def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    sample_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, sample_per_class) # 生成多元正态分布矩阵
    Y0 = np.zeros(sample_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, sample_per_class)
        Y1 = (ci+1)*np.ones(sample_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype = np.float32)
    
    X, Y = shuffle(X0, Y0) # 将所有函数随机排列

    return X, Y


np.random.seed(10) # 定义随机数种子，以确保每一次调用代码生成的随机数相同;如果不设置seed()，则系统将根据时间选择值，每次生成的随机数将不同
num_classes = 2

mean = np.random.randn(num_classes) # 随机生成2个满足正态分布的随机数
cov = np.eye(num_classes) # np.eye()生成对角矩阵 
print(mean) # [1.3315865  0.71527897]
print(cov) # [[1. 0.] [0. 1.]]

X, Y = generate(1000, mean, cov, [3.0], True) # 3.0表明两类的类间差为3,True表明使用非one-hot编码
colors = ['r' if l == 0 else 'b' for l in Y] # 根据Y中的labels设置点的颜色
plt.scatter(X[:, 0], X[:, 1], c = colors) # 绘制散点图
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.savefig("Original 2-Class Data.png")
plt.show()

print(X.shape) # (1000, 2)
print(Y.shape) # (1000,)

"""
参数设置
"""
input_dim = 2 # 设置输入数据的列数
lab_dim = 1 # 设置标签数据的列数

input_features = tf.placeholder(tf.float32, [None, input_dim]) # 设置占位符存储数据，[None, input_dim]表示多维数据但是行数未定
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name = "weight")
b = tf.Variable(tf.zeros([lab_dim]), name = 'bias')

output = tf.nn.sigmoid(tf.matmul(input_features, W) + b) # 使用sigmoid函数
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output)) # 计算交叉熵
ser = tf.square(input_labels - output)

loss = tf.reduce_mean(cross_entropy) # 使用交叉熵计算loss,loss取交叉熵各个维度元素的平均值　
err = tf.reduce_mean(ser) # 使用平方差估计错误率，err取均方差在各个维度的平均值

optimizer = tf.train.AdamOptimizer(0.04) # 使用Adam优化，学习率设置为0.04

train = optimizer.minimize(loss)

"""
构建网络
"""
maxEpochs = 50 # 迭代50次
minibatchSize = 25 # 每个minibatchSize取25条

# 启动Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        # 从minibatchSize取出数据
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize, :]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize], [-1, 1]) # 将y1变形为[-1,1],即列数为1,行数由总数/列数决定
            tf.reshape(y1, [-1, 1])
            _,lossval, outputval,errval = sess.run([train,loss,output,err], feed_dict={input_features: x1, input_labels:y1})
            sumerr += errval
        
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(lossval), "err=", sumerr/np.int32(len(Y)/minibatchSize))

    # 图形显示
    train_X, train_Y = generate(1000, mean, cov, [3.0], True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)

    x = np.linspace(-1,8,200) 
    y = -x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y, label='Fitted line')
    plt.legend()
    plt.savefig("2-Class Data with lines.png")
    plt.show() 
