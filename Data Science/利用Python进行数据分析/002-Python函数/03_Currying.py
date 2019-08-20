"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 函数科里化，参考《利用Python进行数据分析》3.2.5
"""

# 科里化指通过部分参数应用方式从已有函数中衍生出新的函数
def add_number(x, y):
    return x+y

# 方法一
add_five1 = lambda y : add_number(5, y)

# 方法二
from functools import partial
add_five2 = partial(add_number, 5)