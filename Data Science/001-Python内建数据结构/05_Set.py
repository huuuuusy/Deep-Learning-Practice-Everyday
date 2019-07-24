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

"""
常用集合操作
        
a.add(x)
    将元素x加入集合a

a.clear()
    将集合a置空，清空所有元素

a.remove(x)
    从集合a移除某元素

a.pop()
    移除任意元素，空集合则报错

a.union(b)
a | b
    a和b的所有不同元素，并集

a.update(b)
a |= b
    将a的内容设置为a和b的并集

a.intersection(b)
a & b
    a和b同时包含的元素，交集

a.intersection_update(b)
a &= b
    将a的内容设置为a和b的交集

a.difference(b)
a - b
    在a但是不在b的元素

a.difference_update(b)
a -= b
    将a的内容设置为所有在a但是不在b的元素

a.symmetric_difference(b)
a ^ b
    所有在a或b,但是不同时在a,b中的元素

a.symmetric_difference_update(b)
a ^= b
    将a的内容设置为所有在a或b,但是不同时在a,b中的元素

a.issubset(b)
    如果a是b的子集，则返回True

a.issuperset(b)
    如果a包含b,则返回True

a.isdisjoint(b)
    a和b没有交集，则返回True
"""