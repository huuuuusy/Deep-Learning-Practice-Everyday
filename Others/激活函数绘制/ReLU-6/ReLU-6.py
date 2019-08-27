"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： 绘制ReLU曲线
参考： https://segmentfault.com/a/1190000006158803
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False # 用来正常显示负号

# 设置图片大小 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

# ReLU函数,以6为阈值 
x = np.arange(-10, 10)
y = np.where(np.where(x<0, 0, x)<6, np.where(x<0, 0, x), 6)

# 坐标轴显示范围 
plt.xlim(-10,10)
plt.ylim(-10,10)
 
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# tick为刻度线，设置刻度线的位置 
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-10,-6,0,6,10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([-10,-6,6,10])

# 绘图 
plt.plot(x,y,label="ReLU-6",color = "blue")
plt.legend()
plt.savefig("ReLU-6.png")
plt.show()
