"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： Python的lambda函数，参考《利用Python进行数据分析》3.2.4
"""

# lambda是匿名函数
# 以下例子是根据字符串中不同字母的数量对一个字符串集合进行排序
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
print(strings)