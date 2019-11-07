# 005-pandas入门

## 01 pandas数据结构

### DataFrame构造函数的有效输入

|类型|注释|
| :--: |:--: |
|2D ndarray|数据的矩阵，行和列标签是可选参数|
|数组、列表和元组构成的字典|每个序列为DataFrame的一列，所有序列的长度必须相等|
|NumPy结构化/记录化数组|与数组构成的字典一致|
|Series构成的字典|每个值成为一列，每个Series的索引联合形成结果的行索引，也可以显式地传递索引|
|字典构成的字典|每个内部字典成为一列，键联合起来形成结果的行索引|
|字典或者Series构成的列表|列表中每个元素形成DataFrame的一行，字典键或者Series索引联合形成DataFrame的列标签|
|字典或者元组构成的列表|与2D ndarray情况一致|
|其他DataFrame|如果不是显式传递索引，则会使用原DataFrame的索引|
|NumPy MaskedArray|与2D ndarray情况类似，但隐蔽值会在结果的DataFrame中成为NA/缺失值|

### 一些索引对象的方法和属性

|方法|属性|
| :--: |:--: |
|append|将额外的索引对象粘贴到原索引后，产生一个新索引|
|difference|计算两个索引的差集|
|intersection|计算两个索引的交集|
|union|计算两个索引的并集|
|isin|计算表示每一个值是否在传值容器中的布尔数组|
|delete|将位置i的元素删除，并产生新的索引|
|drop|根据传参删除指定索引值，并产生新的索引|
|insert|在位置i插入元素，并产生新的索引|
|is_monotonic|如果索引序列递增则返回True|
|is_unique|如果索引序列唯一则返回True|
|unique|计算索引的唯一值序列|
