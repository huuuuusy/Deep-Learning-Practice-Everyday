"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.37
介绍： np.random.seed()函数学习
参考： https://www.jianshu.com/p/b16b37f6b439
"""

from numpy.random import rand
import numpy as np

# 不使用seed，则生成的随机数不同
a = rand(5)
print('第一次列表a：',a) # [0.96603424 0.6190779  0.07367498 0.72472646 0.40995133]

a = rand(5)
print('第二次列表a：',a) # [0.03259246 0.63893468 0.07146516 0.53433257 0.98120252]

# 使用seed,则生成的随机数相同
np.random.seed(3)
b = rand(5)
print('第一次列表b：',b) # [0.5507979  0.70814782 0.29090474 0.51082761 0.89294695]

np.random.seed(3)
b = rand(5)
print('第二次列表b：',b) # [0.5507979  0.70814782 0.29090474 0.51082761 0.89294695]