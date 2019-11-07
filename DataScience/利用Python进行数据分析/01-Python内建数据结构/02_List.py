"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python列表，参考《利用Python进行数据分析》3.1.2
"""

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
print(b_list) # ['foo', 'bar', 'baz', 'dwarf']

# insert在特定位置插入元素
print('Example 13:')
b_list.insert(1, 'red')
print(b_list) # ['foo', 'red', 'bar', 'baz', 'dwarf']

# pop在特定位置弹出元素
print('Example 14:')
b_list.pop(2)
print(b_list) # ['foo', 'red', 'baz', 'dwarf']

# remove删除第一个符合条件的值
print('Example 15:')
b_list.remove('foo')
print(b_list) # ['red', 'baz', 'dwarf']

# in和not in判断元素是否在list中,返回布尔值
print('Example 16:')
print('dwarf' in b_list) # True 
print('dwarf' not in b_list) # False 

# 用＋连接列表
print('Example 17:')
print([4, None, 'foo'] + [7, 8, (2, 3)]) # [4, None, 'foo', 7, 8, (2, 3)]

# 用extend添加多个列表元素
print('Example 18:')
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)]) 
print(x) # [4, None, 'foo', 7, 8, (2, 3)]

# sort排序
print('Example 19:')
a = [7, 2, 5, 1, 3]
a.sort()
print(a) # [1, 2, 3, 5, 7]
b = ['saw', 'small', 'He', 'foxes', 'six'] 
b.sort(key=len) # 传递一个二级排序key
print(b) # ['He', 'saw', 'six', 'small', 'foxes']

# bisect模块实现二分搜索和已排序列表的插值(bisect本身不检查列表是否已经排序)
print('Example 20:')
import bisect
c = [1, 2, 2, 2, 3, 4, 7]
bisect.bisect(c, 2)
bisect.bisect(c, 5)
bisect.insort(c, 6)
print(c) # [1, 2, 2, 2, 3, 4, 6, 7]

# 切片
print('Example 21:')
seq = [7, 2, 3, 7, 5, 6, 0, 1]
print(seq[1:5]) # [2, 3, 7, 5]
seq[3:4] = [6, 3]
print(seq) # [7, 2, 3, 6, 3, 5, 6, 0, 1]

# 使用默认的起始位置和结束位置
print('Example 22:')
print(seq[:5]) # [7, 2, 3, 6, 3]
print(seq[3:]) # [6, 3, 5, 6, 0, 1]

# 负数索引可以从尾部进行检索
print('Example 23:')
print(seq[-4:]) # [5, 6, 0, 1]
print(seq[-6:-2]) # [6, 3, 5, 6]

# 使用步进值
print('Example 24:')
print(seq[::2]) # 每隔2个数取一个值 # [7, 3, 3, 6, 1]
print(seq[::-1]) # 反转列表　# [1, 0, 6, 5, 3, 6, 3, 2, 7]