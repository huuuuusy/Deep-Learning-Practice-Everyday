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

    X0 = np.random.multivariate_normal(mean, cov, sample_per_class)
    Y0 = np.zeros(sample_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, sample_per_class)
        Y1 = (ci+1)*np.ones(sample_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype = np.float32)
    
    X, Y = shuffle(X0, Y0)

    return X, Y


input_dim = 2
np.random.seed(10) # 定义随机数种子，以确保每一次调用代码生成的随机数相同;如果不设置seed()，则系统将根据时间选择值，每次生成的随机数将不同
num_classes = 2

mean = np.random.rand(num_classes)
cov = np.eye(num_classes)
print(mean)
print(cov)