"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 熟悉pandas数据结构，参考《利用Python进行数据分析》5.1
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

"""
Series
    包含一个值序列和一个数据标签
    一维的数组型对象
"""
# 默认的index是从0开始的计数
print('Example 1:')
obj = pd.Series([4, 7, -5, 3]) 
print(obj)
#0    4
#1    7
#2   -5
#3    3
#dtype: int64

print('Example 2:')
print(obj.values) # [ 4  7 -5  3]
print(obj.index) # RangeIndex(start=0, stop=4, step=1) 类似于range(4)

print('Example 3:')
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
#d    4
#b    7
#a   -5
#c    3
#dtype: int64
print(obj2.index) # Index(['d', 'b', 'a', 'c'], dtype='object')

print('Example 4:')
print(obj2['a']) # -5
obj2['d'] = 6
print(obj2[['c', 'a', 'd']])
#c    3
#a   -5
#d    6
#dtype: int64

# 使用布尔值数组进行过滤
print('Example 5:')
print(obj2[obj2 > 0])
#d    6
#b    7
#c    3
#dtype: int64

# 数组和标量相乘
print('Example 6:')
print(obj2 * 2)
#d    12
#b    14
#a   -10
#c     6
#dtype: int64

# 应用数学函数
print('Example 7:')
print(np.exp(obj2))
#d     403.428793
#b    1096.633158
#a       0.006738
#c      20.085537
#dtype: float64

print('Example 8:')
print('b' in obj2) # True
print('e' in obj2) # False

# 可以使用字典生成一个Series
print('Example 9:')
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
print(obj3)
#Ohio      35000
#Texas     71000
#Oregon    16000
#Utah       5000
#dtype: int64

# 可以指定索引值生成Series
# NaN表示缺失值
print('Example 10:')
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
print(obj4)
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#dtype: float64

# 判断是否有缺失值
# 使用pd.isnull或pd.notnull方法进行判别
print('Example 11:')
print(pd.isnull(obj4)) # 等价于obj4.isnull()
#California     True
#Ohio          False
#Oregon        False
#Texas         False
#dtype: bool
print(pd.notnull(obj4)) # 等价于obj4.notnull()
#California    False
#Ohio           True
#Oregon         True
#Texas          True
#dtype: bool

# 自动对齐
print('Example 12:')
print(obj3)
#Ohio      35000
#Texas     71000
#Oregon    16000
#Utah       5000
#dtype: int64
print(obj4)
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#dtype: float64
print(obj3 + obj4) # 多个Series在操作时会自动对齐索引相同的元素
#California         NaN
#Ohio           70000.0
#Oregon         32000.0
#Texas         142000.0
#Utah               NaN
#dtype: float64

# Series对象自身和其索引都有name属性
print('Example 13:')
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)
#state
#California        NaN
#Ohio          35000.0
#Oregon        16000.0
#Texas         71000.0
#Name: population, dtype: float64

# Series的索引可以通过按照位置赋值的方式改变
print('Example 14:')
print(obj)
#0    4
#1    7
#2   -5
#3    3
#dtype: int64
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)
#Bob      4
#Steve    7
#Jeff    -5
#Ryan     3
#dtype: int64

"""
DataFrame
    表示矩阵的数据表
    包含已经排序的列集合，每一列可以是不同类型的值(数值，字符串，布尔值等)
    有行索引和列索引，相当于一个共享相同索引的Series的字典
    数据被存储为一个以上的二维块
"""

# 利用包含等长的列表或者NumPy数组的字典创建DataFrame
print('Example 15:')
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame)
#    state  year  pop
#0    Ohio  2000  1.5
#1    Ohio  2001  1.7
#2    Ohio  2002  3.6
#3  Nevada  2001  2.4
#4  Nevada  2002  2.9
#5  Nevada  2003  3.2

# head方法将选出头部的前5行
print('Example 16:')
print(frame.head())
#    state  year  pop
#0    Ohio  2000  1.5
#1    Ohio  2001  1.7
#2    Ohio  2002  3.6
#3  Nevada  2001  2.4
#4  Nevada  2002  2.9

# DataFrame可以使用columns指定列的顺序　
print('Example 17:')
print(pd.DataFrame(data, columns=["year", "state", "pop"]))
#   year   state  pop
#0  2000    Ohio  1.5
#1  2001    Ohio  1.7
#2  2002    Ohio  3.6
#3  2001  Nevada  2.4
#4  2002  Nevada  2.9
#5  2003  Nevada  3.2

print('Example 18:')
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four', 'five', 'six'])
print(frame2) # 当出现缺失值时，也使用NaN进行填补
#       year   state  pop debt
#one    2000    Ohio  1.5  NaN
#two    2001    Ohio  1.7  NaN
#three  2002    Ohio  3.6  NaN
#four   2001  Nevada  2.4  NaN
#five   2002  Nevada  2.9  NaN
#six    2003  Nevada  3.2  NaN
print(frame2.columns)
# Index(['year', 'state', 'pop', 'debt'], dtype='object')

# 可以类似于字典标记或者属性标记的方式检索DataFrame,生成Series
print('Example 19:')
print(frame2['state']) # 对于任意列名均有效
#one        Ohio
#two        Ohio
#three      Ohio
#four     Nevada
#five     Nevada
#six      Nevada
#Name: state, dtype: object
print(frame2.year) # 只在列名是有效的Python变量名时才有效
#one      2000
#two      2001
#three    2002
#four     2001
#five     2002
#six      2003
#Name: year, dtype: int64

print('Example 20:')


print('Example 21:')


print('Example 22:')


print('Example 23:')


print('Example 24:')


print('Example 25:')


print('Example :')



