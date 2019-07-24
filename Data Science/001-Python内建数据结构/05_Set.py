"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python集合，参考《利用Python进行数据分析》3.1.5
"""

"""
集合
"""
# 集合使用set()函数，其无序且元素唯一
print('Example 41:')
print(set([2, 2, 2, 1, 3, 3]))

# 集合并集
print('Example 42:')
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
print(a.union(b))
print(a | b)

# 集合交集
print('Example 43:')
print(a.intersection(b))
print(a & b)

# 集合元素不可变，如果想要包含列表型元素，需要转换为元组
print('Example 44:')
my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
print(my_set)

# 判断一个集合是否为另一个集合的子集或者超集
print('Example 45:')
a_set = {1, 2, 3, 4, 5}
print({1, 2, 3}.issubset(a_set))
print(a_set.issuperset({1, 2, 3}))

# 判断集合是否相等
print('Example 46:')
print({1, 2, 3} == {3, 2, 1})

