"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python的推导式，参考《利用Python进行数据分析》3.1.6
"""

"""
列表推导式
[expr for val in collection if condition]
"""
print('Example 47:')
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
print([x.upper() for x in strings if len(x) > 2]) #  ['BAT', 'CAR', 'DOVE', 'PYTHON']

"""
字典推导式
dict_comp = {key-expr: value-expr for value in collection if condition}
"""
print('Example 48:')
loc_mapping = {val : index for index, val in enumerate(strings)}
print(loc_mapping) # {'a': 0, 'as': 1, 'bat': 2, 'car': 3, 'dove': 4, 'python': 5}

"""
集合推导式
set_comp = {expr for val in collection if condition}
"""
print('Example 49:')
unique_lengths = {len(x) for x in strings}
print(unique_lengths) # {1, 2, 3, 4, 6}

"""
嵌套列表推导式,其中for的嵌套和for循环完全一致
"""
print('Example 50:')
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
result = [name for names in all_data for name in names
          if name.count('e') >= 2]
print(result) # ['Steven']

print('Example 51:')
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
print(flattened) # [1, 2, 3, 4, 5, 6, 7, 8, 9]
