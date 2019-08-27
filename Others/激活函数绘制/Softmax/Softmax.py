"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： 绘制Softmax曲线
"""

import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import numpy as np
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.linspace(-10, 10, 200)
y = softmax(x)

plt.xlim(-10, 10)
plt.ylim(0, 0.1)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.set_xticks([-10, -5, 0, 5, 10])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([0.02, 0.04, 0.06, 0.08, 0.10])

plt.plot(x,y,label='Softmax', color='blue')
plt.legend()
plt.savefig("Softmax.png")
plt.show()