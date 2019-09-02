"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 欠拟合、过拟合与正则化的验证
"""

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

"""
曲线拟合
"""
# 目标函数为sin(x)
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式函数
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residual_func(p,x,y):
    return  fit_func(p,x) -y

# 设置10个点
x = np.linspace(0,1,10)
x_points = np.linspace(0,1,1000)

# 加上正态分布噪音的目标函数值
y = [np.random.normal(0,0.1) + y1 for y1 in real_func(x)]

def fitting(M=0):
    """
    M为多项式的次数
    """
    # 随机初始化多项式参数　
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residual_func, p_init, args=(x,y))
    print('Fitting Parameters:',p_lsq[0])
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.title('M='+ str(M))
    plt.legend()
    plt.savefig('M='+ str(M) +'.png')
    plt.show()
    return p_lsq

"""
M=0,1:欠拟合
M=3:合适
M=9:过拟合
"""
# M=0
p_lsq_0 = fitting(M=0)

# M=1
p_lsq_1 = fitting(M=1)

# M=3
p_lsq_3 = fitting(M=3)

# M=9
p_lsq_9 = fitting(M=9)

"""
正则化
"""
regularization = 0.0001

def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret,np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret

# 最小二乘法,加正则化项
p_init = np.random.rand(9 + 1)
p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))

plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
plt.plot(x_points,fit_func(p_lsq_regularization[0], x_points),label='regularization')
plt.plot(x, y, 'bo', label='noise')
plt.title('Regularization')
plt.legend()
plt.savefig('Regularization.png')
plt.show()