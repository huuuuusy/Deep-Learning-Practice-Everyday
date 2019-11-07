"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 熟悉pandas基本功能，参考《利用Python进行数据分析》5.2
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

"""
重建索引
"""
print('Example 1:')
obj = pd.Series([4.5, 7.3, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
print(obj)
#d    4.5
#b    7.3
#a   -5.3
#c    3.6
#dtype: float64

# reindex方法将按照新的索引排序，如果索引的值不存在，将填入缺失值
print('Example 2:')
obj2 = obj.reindex(['a', 'b', 'c','d','e'])
print(obj2)
#a   -5.3
#b    7.3
#c    3.6
#d    4.5
#e    NaN
#dtype: float64

# 可以使用ffill等方法在重建索引时进行差值
# ffill将值进行前向填充
print('Example 3:')
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
print(obj3)
#0      blue
#2    purple
#4    yellow
#dtype: object
print(obj3.reindex(range(6), method='ffill'))
#0      blue
#1      blue
#2    purple
#3    purple
#4    yellow
#5    yellow
#dtype: object

print('Example 4:')

print('Example 5:')

print('Example 6:')

print('Example 7:')

print('Example 8:')

print('Example 9:')

print('Example 10:')

print('Example 11:')

print('Example 12:')

print('Example 13:')

print('Example 14:')

print('Example 15:')

print('Example 16:')

print('Example 17:')

print('Example 18:')

print('Example 19:')

print('Example 20:')

print('Example 21:')

print('Example 22:')

print('Example 23:')

print('Example 24:')

print('Example 25:')

print('Example 26:')

print('Example 27:')

print('Example :')


