"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python内置的数据结构，参考《利用Python进行数据分析》3.1
"""

"""
元组
"""
print('Tuple:')

# 创建
print('Example 1:')
tup = 4, 5, 6
print(tup)

print('Example 2:')
nested_tup = (4, 5, 6), (7, 8)
print(nested_tup)

# 元组不可变，但是若元组上某个对象可以改变，则可以进行内部修改
print('Example3:')
tup = tuple(['foo', [1, 2], True])
tup[1].append(3)
print(tup)

# 用+连接元组
print('Example 4:')
print((4, None, 'foo') + (6, 0) + ('bar',))

# 用*拷贝元组
print('Example ５:')
print(('foo', 'bar') * 4)

# 拆包
print('Example 6:')
tup = (4, 5, 6)
a, b, c = tup
print(b)

print('Example 7:')
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
print(d)

# 交换
print('Example 8:')
a, b = 1, 2
b, a = a, b
print(a, b)

# 遍历元组
print('Example 9:')
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c))

# 采集元素
print('Example 10:')
values = 1, 2, 3, 4, 5
a, b, *rest = values
print(a, b)
print(rest)

# 计数
print('Example 11:')
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)

"""
列表
"""
print('List:')
a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)

# append在尾部添加元素
print('Example 12:')
b_list.append('dwarf')
print(b_list)

# insert在特定位置插入元素
print('Example 13:')
b_list.insert(1, 'red')
print(b_list)

# pop在特定位置弹出元素
print('Example 14:')
b_list.pop(2)
print(b_list)

# remove删除第一个符合条件的值
print('Example 15:')
b_list.remove('foo')
print(b_list)

# in和not in判断元素是否在list中,返回布尔值
print('Example 16:')
print('dwarf' in b_list)
print('dwarf' not in b_list)

# 用＋连接列表
print('Example 17:')
print([4, None, 'foo'] + [7, 8, (2, 3)])

# 用extend添加多个列表元素
print('Example 18:')
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
print(x)

# sort排序
print('Example 19:')
a = [7, 2, 5, 1, 3]
a.sort()
print(a)
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len) # 传递一个二级排序key
print(b)

# bisect模块实现二分搜索和已排序列表的差值(bisect本身不检查列表是否已经排序)
print('Example 20:')
import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)
bisect.bisect(c, 5)
bisect.insort(c, 6)
print(c)

# 切片
print('Example 21:')
seq = [7, 2, 3, 7, 5, 6, 0, 1]
print(seq[1:5])
seq[3:4] = [6, 3]
print(seq)

# 使用默认的起始位置和结束位置
print('Example 22:')
print(seq[:5]) 
print(seq[3:])

# 负数索引可以从尾部进行检索
print('Example 23:')
print(seq[-4:])
print(seq[-6:-2])

# 使用步进值
print('Example 24:')
print(seq[::2]) # 每隔2个数取一个值
print(seq[::-1]) # 反转列表　

"""
内建函数
enumerate()
sorted()
zip()
reversed()
"""
# enumerate()返回(index, value)元组序列
print('Example 25:')
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i
print(mapping)

# sorted()返回根据任一序列中的元素新建的已排序列表
print('Example 26:')
print(sorted([7, 1, 2, 6, 0, 3, 2]))
print(sorted('horse race'))

# zip()将列表，元组或者其他元素配对，新建一个由元组构成的列表
print('Example 27:')
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zipped = zip(seq1, seq2)
print(list(zipped))

# zip()可以处理长度不同的序列，生成列表长度由最短序列决定
print('Example 28:')
seq3 = [False, True]
print(list(zip(seq1, seq2, seq3)))

# 当同时遍历多个序列时，zip()可以和enumerate()同时使用
print('Example 29:')
for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('{0}: {1}, {2}'.format(i, a, b))

# 巧妙用法：使用zip()拆分序列
print('Example 30:')
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'),
            ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
print(first_names)
print(last_names)

# reversed()将序列元素倒序排列
print('Example 31:')
print(list(reversed(range(10))))

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


print('Example 41:')

print('Example 42:')

print('Example 43:')


print('Example 44:')

print('Example 45:')

print('Example 46:')

print('Example 47:')
