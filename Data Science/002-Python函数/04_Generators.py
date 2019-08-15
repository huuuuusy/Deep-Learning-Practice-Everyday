"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 函数生成器，参考《利用Python进行数据分析》3.2.6
"""

"""
生成器表达式
将生成器表达式和函数相结合，可以直接生成值并且计算结果
"""
print('Example 1:')
print(sum(x ** 2 for x in range(100))) # 328350
print(dict((i, i **2) for i in range(5))) # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

"""
itertools模块
"""
print('Example 2:')
import itertools

first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))
# A ['Alan', 'Adam'] W ['Wes', 'Will'] A ['Albert'] S ['Steven']