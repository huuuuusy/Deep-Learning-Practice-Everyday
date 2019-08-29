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

def onehot(y, start, end):
    a = np.linspace(start, end-1, end-start)
    b = np.reshape(a, [-1,1]).astype(np.int32)
    OneHotEncoder().fit(b)
    c = OneHotEncoder().transform(y).toarray()
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
input_dim = 2

X,Y = generate2(2000, num_classes, [[3.0],[3.0,0]], False)
aa = [np.argmax(l) for l in Y]


