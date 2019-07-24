"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： Python的函数基本用法，参考《利用Python进行数据分析》3.2.3
"""

"""
清洗states列表的几种方法
"""
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
          'south   carolina##', 'West virginia?']

# 方法一
print('Method 1:')
import re

def clean_strings1(strings):
    result = []
    for value in strings:
        value = value.strip() # 去掉空格
        value = re.sub('[!#?]', '', value) # 去掉标点
        value = value.title() # 调整大小写
        result.append(value)
    return result

print(clean_strings1(states))

# 方法二
print('Method 2:')
def remove_punctuation(value):
    return re.sub('[!#?]', '', value) 

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings2(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result

print(clean_strings2(states, clean_ops))

# 方法三
print('Method ３:')
for x in map(remove_punctuation, states):
    print(x)