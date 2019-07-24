"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python内建函数，参考《利用Python进行数据分析》3.1.3
"""

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
