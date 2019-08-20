"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python元组，参考《利用Python进行数据分析》3.1.1
"""

"""
元组
"""
print('Tuple:')

# 创建
print('Example 1:')
tup = 4, 5, 6
print(tup) # (4, 5, 6)

print('Example 2:')
nested_tup = (4, 5, 6), (7, 8) # ((4, 5, 6), (7, 8))
print(nested_tup)

# 元组不可变，但是若元组上某个对象可以改变，则可以进行内部修改
print('Example3:')
tup = tuple(['foo', [1, 2], True])
tup[1].append(3)
print(tup) # ('foo', [1, 2, 3], True)

# 用+连接元组
print('Example 4:')
print((4, None, 'foo') + (6, 0) + ('bar',)) # (4, None, 'foo', 6, 0, 'bar')

# 用*拷贝元组
print('Example ５:')
print(('foo', 'bar') * 4) # ('foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar')

# 拆包
print('Example 6:')
tup = (4, 5, 6)
a, b, c = tup
print(b) # 5

print('Example 7:')
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
print(d) # 7

# 交换
print('Example 8:')
a, b = 1, 2
b, a = a, b
print(a, b) # 2 1

# 遍历元组
print('Example 9:')
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print('a={0}, b={1}, c={2}'.format(a, b, c)) # a=1, b=2, c=3 a=4, b=5, c=6 a=7, b=8, c=9

# 采集元素
print('Example 10:')
values = 1, 2, 3, 4, 5
a, b, *rest = values
print(a, b) # 1 2 
print(rest) # [3, 4, 5]

# 计数
print('Example 11:')
a = (1, 2, 2, 2, 3, 4, 2)
print(a.count(2)) # 4
