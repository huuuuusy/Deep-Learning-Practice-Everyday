"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

使用简单的DNN模型对Jena温度数据集进行处理
"""

"""
Part 1. 数据准备
http://www.bgc-jena.mpg.de/wetter/
"""

import os

data_dir = '/home/hu/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

# 读取数据
f = open(fname)
data = f.read()
f.close()

# 处理数据
# headers是数据的第一行，是每列数据的名字，需要单独提取 
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(lines[0])
print(len(lines))

"""
Part 2. 解析数据
"""

import numpy as np 

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
    
print(float_data.shape)
print(float_data[0])

# 绘制气温的变化图，可以看出气温随时间变化的周期性
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()
