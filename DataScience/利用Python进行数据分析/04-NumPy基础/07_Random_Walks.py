"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用NumPy进行随机漫步，参考《利用Python进行数据分析》4.7
"""
import numpy as np
import matplotlib.pyplot as plt

"""
纯Python的随机漫步
"""
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.figure()
plt.plot(walk[:100])
plt.show()

"""
NumPy随机漫步
"""
nsteps = 1000 
draws = np.random.randint(0,  2, size = nsteps) # 在[0,2)的范围内从低到高抽取1000个随机整数,即随机抽0或1
steps = np.where(draws > 0, 1, -1) # 将抽到1的位置保持为1,0变为-1
walks = steps.cumsum() # 进行累和

plt.figure()
plt.plot(walk[:100])
plt.show()
