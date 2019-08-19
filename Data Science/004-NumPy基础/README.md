# 004-NumPy基础

## 01 ndarray

### 数组生成函数

|操作|内容|
| :--: |:--: |
|arrray|将输入数据转换为ndarray;如果不是显式指明数据类型，将自动推断；默认复制所有数据|
|asarray|将输入数据转换为ndarray|
|arange|range的数组版本，返回一个数组|
|ones|根据给定形状和数据类型生成全1数组|
|ones_like|根据给定的数组生成形状一致的全1数组|
|zeros|根据给定形状和数据类型生成全0数组|
|zeros_like|根据给定的数组生成形状一致的全0数组|
|empty|根据给定形状和数据类型生成没有初始化的空数组|
|empty_like|根据给定的数组生成形状一致的没有初始化的空数组|
|full|根据给定形状和数据类型生成指定数值的数组|
|full_like|根据给定的数组生成形状一致的指定数值的数组|
|eye, identity|生成一个N*N特征矩阵，对角线位置是1，其余位置为0|

## 02 Universal Functions

### 一元通用函数

|操作|内容|
| :--: |:--: |
|abs,fabs|逐元素计算整数、浮点数或复数的绝对值|
|sqrt|计算每个元素的平方根, = arr**0.5|
|square|计算每个元素的平方, = arr**2|
|exp|计算每个元素的自然指数值|
|log,log10,log2,log1p|分别对应自然对数、以10为底对数、以2为底对数、log(1+x)|
|sign|计算每个元素的符号值，1(正数)，0(0), -1(负数)|
|ceil|计算每个元素的最高整数值|
|floor|计算每个元素的最小整数值|
|rint|将元素保留到整数位，并保留dtype|
|modf|分别将数组的小数形式和整数形式按数组形式返回|
|isnan|返回数组中元素是否是一个NaN,形式为布尔值|
|isfinite,isinf|分别返回数组中的元素是否有限、是否无限，形式为布尔值数组|
|cos,cosh,sin,sinh,tan,tanh|常规双曲三角函数|
|arccos,arccosh,arcsin,arcsinh,arctan,arctanh|反三角函数|
|logical_not|对数组元素按位取反, = ~ arr|

### 二元通用函数

|操作|内容|
| :--: |:--: |
|add|将数组对应元素相加|
|subtract|在第二个数组中，将第一个数组中包含的元素去除|
|multiply|将数组对应元素相乘|
|divide,floor_divide|除或整除|
|power|将第二个数组元素作为第一个数组中元素的幂次方|
|maximum,fmax｜逐元素计算最大值，fmax忽略NaN｜
|minimum,fmin｜逐元素计算最小值，fmin忽略NaN｜
|mod|逐元素求模|
|copysign|将第一个数组的符号值改为第二个数组的符号值|
|greater,greater_equal,less,less_equal,equal,not_equal|逐元素比较，返回布尔值数组｜
|logical_and,logical_or,logical_xor|逐元素逻辑操作|

## 03 Array-Oriented Programming with Arrays

### 基础数组统计方法

|操作|内容|
| :--: |:--: |
|sum|沿着轴向计算所有元素的累和，0长度的数组累和为0|
|mean|数学平均,0长度的数组的数学平均值为NaN|
|std, var|标准差和方差，可以调整自由度，默认分母是n|
|min, max|最小值和最大值|
|argmin, argmax|最小值和最大值的位置|
|cumsum|从0开始的元素的累积和|
|cumprod|从1开始的元素的累积积|

### 数组的集合操作

|操作|内容|
| :--: |:--: |
|unique(x)|计算x的唯一值并排序|
|intersect1d(x,y)|计算x和y的交集并排序|
|union1d(x,y)|计算x和y的并集并排序|
|in1d(x,y)|计算x中的元素是否在y中并返回一个布尔值数组|
|setdiff1d(x,y)|差集，在x中但是不在y中的元素|
|setxor1d(x,y)|异或集，在x或者y中，但是不属于x与y交集的元素|

## 05_ Linear Algebra

### 常用numpy.linalg函数

|操作|内容|
| :--: |:--: |
|diag|将一个方阵的对角元素作为一维数组返回，或者将一维数组转换成一个方阵,并且非对角线上有零点|
|dot|矩阵点乘|
|trace|计算对角线元素和|
|det|计算矩阵行列式|
|eig|计算方阵的特征值和特征向量|
|inv|计算方阵的逆矩阵|
|pinv|计算矩阵的Moore-Penrose伪逆|
|qr|计算QR分解|
|svd|计算奇异值分解(SVD)|
|solve|求解线性系统Ax=b,其中A是方阵|
|lstsq|计算Ax=b的最小二乘解|

## 06_Pseudorandom Number Generation

### numpy.random中部分函数列表

|操作|内容|
| :--: |:--: |
|seed|向随机数生成器传递随机状态种子|
|permutation|返回一个序列的随机排列，或者返回一个乱序的整数范围序列|
|shuffle|随机排列一个序列|
|rand|从均匀分布中抽取样本|
|randint|根据给定的从低到高的范围抽取随机数|
|randn|从均值0方差1的正态分布中抽取样本|
|binomial|从二项分布中抽取样本|
|normal|从正态（高斯）分布中抽取样本|
|beta|从beta分布中抽取样本|
|chisquare|从卡方分布中抽取样本|
|gamma|从gamma分布中抽取样本|
|uniform|从均匀[0,1)分布中抽取样本|
