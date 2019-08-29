"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
框架： TensorFlow-GPU == 1.13.1
介绍： 线性多分类问题分析
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

def onehot(y,start,end):
    ohe = OneHotEncoder() # 使用OneHotEncoder时，要先指定ohe=OneHotEncoder(),直接使用OneHotEncoder.fit()会出错，ohe.fit()正确
    a = np.linspace(start,end-1,end-start)
    b =np.reshape(a,[-1,1]).astype(np.int32)
    ohe.fit(b)
    c=ohe.transform(y).toarray()  
    return c

def generate2(sample_size, num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)

    samples_per_class = int(sample_size/num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression==False:
        Y0 = np.reshape(Y0,[-1,1]) # Y0被reshape为一个行数不定，列数为1的矩阵
        Y0 = onehot(Y0.astype(np.int32),0,num_classes) # 对Y0进行onehot编码

    X, Y = shuffle(X0, Y0)
    return X,Y

np.random.seed(10)
num_classes = 3

X, Y = generate2(2000,num_classes,[[3.0],[3.0,0]],False)
aa = [np.argmax(l) for l in Y] # aa[:20]--> [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 1, 0, 1, 2, 2, 0, 1]
colors =['r' if l == 0 else 'b' if l==1 else 'y' for l in aa[:]]

plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.savefig('Original 3-Class Data.png')
plt.show()

"""
参数设置
"""

input_dim = 2
lab_dim = num_classes

input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name = 'weight')
b = tf.Variable(tf.zeros([lab_dim]), name = 'bias')

output = tf.matmul(input_features, W) + b
z = tf.nn.softmax(output)

a1 = tf.argmax(tf.nn.softmax(output), axis = 1) # 按行找出最大索引，生成数组
b1 = tf.argmax(input_labels, axis = 1)
err = tf.count_nonzero(a1 - b1) # 两个数组相减，不为0的个数就是错误的个数

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(0.04) # Adam优化器可以动态调节梯度，收敛快
train = optimizer.minimize(loss)

"""
构建网络
"""

maxEpochs = 50
miniBatchSize = 25

# 启动Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        for i in range(np.int32(len(Y)/miniBatchSize)):
            x1 = X[i*miniBatchSize:(i+1)*miniBatchSize, :]
            y1 = Y[i*miniBatchSize:(i+1)*miniBatchSize, :]

            _, lossval, outputval, errval = sess.run([train, loss, output, err], feed_dict={input_features:x1, input_labels:y1})
            sumerr += errval/miniBatchSize

        print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(lossval),"err=",sumerr/(np.int32(len(Y)/miniBatchSize)))

    """
    绘制线性逻辑回归得到的直线
    """
    train_X, train_Y = generate2(2000,num_classes, [[3.0],[3.0,0]],False)
    aa = [np.argmax(l) for l in train_Y]        
    colors =['r' if l == 0 else 'b' if l==1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    
    x = np.linspace(-3,8,200) 

    y=-x*(sess.run(W)[0][0]/sess.run(W)[1][0])-sess.run(b)[0]/sess.run(W)[1][0]
    plt.plot(x,y, label='first line',lw=3)

    y=-x*(sess.run(W)[0][1]/sess.run(W)[1][1])-sess.run(b)[1]/sess.run(W)[1][1]
    plt.plot(x,y, label='second line',lw=2)

    y=-x*(sess.run(W)[0][2]/sess.run(W)[1][2])-sess.run(b)[2]/sess.run(W)[1][2]
    plt.plot(x,y, label='third line',lw=1)
    plt.legend()
    plt.savefig("3-Class Data with lines.png")
    plt.show() 
    
    """
    绘制线性逻辑回归得到的区域分割图
    """
    train_X, train_Y = generate2(2000,num_classes,  [[3.0],[3.0,0]],False)
    aa = [np.argmax(l) for l in train_Y]        
    colors =['r' if l == 0 else 'b' if l==1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)    
    
    nb_of_xs = 200
    xs1 = np.linspace(-3, 8, num=nb_of_xs)
    xs2 = np.linspace(-3, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2) # 创建grids
    # 初始化分类面板
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i,j] = sess.run(a1, feed_dict={input_features: [[ xx[i,j], yy[i,j] ]]} )
    
    # 创建colormap
    cmap = ListedColormap([
            colorConverter.to_rgba('r', alpha=0.30),
            colorConverter.to_rgba('b', alpha=0.30),
            colorConverter.to_rgba('y', alpha=0.30)])
    # 绘制colormap和分割线
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.savefig("3-Class Data with colormap.png")
    plt.show()
