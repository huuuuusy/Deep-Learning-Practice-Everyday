"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 介绍Python文件操作，参考《利用Python进行数据分析》3.３
"""

"""
常见文件操作

read([size])
    将文件数据作为字符串返回，size控制读取字节数

readlines([size])
    返回文件中行内容列表

write(str)
    将字符串写入文件

writelines(strings)
    将字符串序列写入文件

close()
    关闭文件

flush()
    将内部I/O缓冲器内容刷新到硬盘

seek(pos)
    移动到指定位置

tell()
    返回当前文件位置

closed
    如果文件已经关闭，返回True
"""

"""
Python文件模式

r   只读模式
w   只写模式，创建新文件，覆盖同名文件
x   只写模式，创建新文件，存在同名文件时创建失败
a   添加到已经存在的文件，不存在则创建
r+  读写模式
b   二进制文件模式
t   文本文件模式，将字节自动解码为Unicode
"""