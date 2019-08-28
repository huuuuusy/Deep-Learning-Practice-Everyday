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
