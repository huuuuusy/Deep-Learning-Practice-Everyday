# 001-Python内建数据结构

## 05 Set

### 常用集合元素操作

|操作|内容|
| :--: |:--: |
|a.add(x)|将元素x加入集合a|
|a.clear()|将集合a置空，清空所有元素|
|a.remove(x)|从集合a移除某元素|
|a.pop()|移除任意元素，空集合则报错|
|a.union(b)或者a \| b|a和b的所有不同元素，并集|
|a.update(b)或者a \|= b|将a的内容设置为a和b的并集|
|a.intersection(b)或者a & b|a和b同时包含的元素，交集|
|a.intersection_update(b)或者a &= b|将a的内容设置为a和b的交集|
|a.difference(b)或者a - b|在a但是不在b的元素|
|a.difference_update(b)或者a -= b|将a的内容设置为所有在a但是不在b的元素|
|a.symmetric_difference(b)或者a ^ b|所有在a或b,但是不同时在a,b中的元素|
|a.symmetric_difference_update(b)或者a ^= b|将a的内容设置为所有在a或b,但是不同时在a,b中的元素|
|a.issubset(b)|如果a是b的子集，则返回True|
|a.issuperset(b)|如果a包含b,则返回True|
|a.isdisjoint(b)|a和b没有交集，则返回True|
