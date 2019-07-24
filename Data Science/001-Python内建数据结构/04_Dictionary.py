"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python字典，参考《利用Python进行数据分析》3.1.4
"""

"""
字典
"""
print('Example 32:')
empty_dict = {}
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
print(d1)
d1[7] = 'an integer' # 插入元素
print(d1)
print(d1['b']) # 检索元素

# 使用in检查元素是否在字典的键中
print('Example 33:')
print('b' in d1)

# 使用del()删除值
print('Example 34:')
d1[5] = 'some value'
d1['dummy'] = 'another value'
print(d1)
del d1[5]
print(d1)

# 使用pop()删除值,pop()会返回被删除的值并删除键
print('Example 35:')
ret = d1.pop('dummy')
print(ret)
print(d1)

# keys()和values()分别提供键、值的迭代器
# 返回的键或者值没有特定顺序，但是会保持键－值的对应关系
print('Example 36:')
print(list(d1.keys()))
print(list(d1.values()))

# update()可以将两个字典合并
print('Example 37:')
d1.update({'b' : 'foo', 'c' : 12})
print(d1)

# 从序列生成字典
print('Example 38:')
mapping = dict(zip(range(5), reversed(range(5))))
print(mapping)

# 使用默认值构建字典
print('Example 39:')
words = ['apple', 'bat', 'bar', 'atom', 'book']
from collections import defaultdict 
by_letter = defaultdict(list) 
for word in words: 
    by_letter[word[0]].append(word)

# 字典的键必须是不可变对象
# 如果想用列表作为键，必须先哈希化,将其转化为元组
print('Example 40:')
d = {}
d[tuple([1, 2, 3])] = 5
print(d)
