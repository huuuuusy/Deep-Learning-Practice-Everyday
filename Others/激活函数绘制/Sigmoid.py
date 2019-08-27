"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： 绘制Sigmoid曲线
参考： https://segmentfault.com/a/1190000006158803
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

x = np.arange(-7, 7, 0.1)
y = 1/(1+np.exp(-x))

plt.xlim(-6.2, 6.2)
plt.ylim(0, 1.2)
 
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
 
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-6, -3, 0, 3, 6])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([0, 0.5, 1])
 
plt.plot(x,y,label="Sigmoid",color = "blue")
plt.legend()
plt.savefig("Sigmoid.png")
plt.show()
