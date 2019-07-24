"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： NumPy操作，参考《利用Python进行数据分析》4.1
"""

import numpy as np

"""
生成ndarry
"""
# 从list生成ndarray
print('Example 1:')
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
print(arr1.dtype)

# 长度相同的List将生成多维ndarray
print('Example 2:')
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim)
print(arr2.shape)
print(arr2.dtype)

# 创建
print('Example 3:')
print(np.zeros(10))
print(np.ones((3,6)))
print(np.empty((2,3,2))) # 创建没有初始化值的数组

# arange()函数生成数组，类似于range
print('Example 4:')
print(np.arange(15))

"""
ndarray数据类型
"""
# dtype指定数据类型
print('Example 5:')
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)

# astype转化数据类型
print('Example 6:')
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
float_arr = arr.astype(np.float64)
print(float_arr.dtype)

# 使用其他数组的dtype来转化数据类型
print('Example 7:')
int_array = np.arange(10)
print(int_array)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
print(int_array.astype(calibers.dtype))

"""
NumPy数组算术
"""
# 任意两个同尺寸的数组之间的算术操作都应用了逐元素的操作方式
print('Example 8:')
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)
print(arr * arr) # 逐元素相乘
print(arr - arr)

# 带有标量计算的算术操作，会把计算参数传递给数组的每一个元素
print('Example 9:')
print(1 / arr)
print(arr ** 0.5)

# 同尺寸数组之间比较，会产生一个布尔值
print('Example 10:')
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print(arr2)
print(arr2 > arr) # 逐元素比较

"""
基础索引与切片
"""
print('Example 11:')
arr = np.arange(10)
print(arr)
print(arr[5:8]) # [5 6 7]
# 注意:修改切片时，原数组也会被改变，说明NumPy的切片是直接在原数组上进行修改，而非对原数组的拷贝
# 这一点与python自带的list等数据结构完全不同，一定要区分！！！
arr[5:8] = 12
print(arr) # [ 0  1  2  3  4 12 12 12  8  9]
# 注意：即时使用另一个变量名，只要没有复制切片，修改切片值时仍然会影响原数组
# 如果想不影响原数组，需要使用arr[5:8].copy()进行操作
arr_slice = arr[5:8]
arr_slice[1] = 12345
print(arr) # [    0     1     2     3     4    12 12345    12     8     9]

"""
切片索引
"""
print('Example 12:')
print(arr) # [    0     1     2     3     4    12 12345    12     8     9]
print(arr[1:6]) # [ 1  2  3  4 12]

# 对于高维数组，arr2d[:2]指按照轴0(行)进行切片
print('Example 13:')
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d)
print(arr2d[:2]) # [[1 2 3][4 5 6]]

# arr2d[:2, :1]指定行切片和列切片
print('Example 14:')
print(arr2d[:2, 1:]) # [[2 3][5 6]]

"""
布尔索引
使用布尔索引时，总是生成数据的拷贝
"""
print('Example 15:')
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(names)
print(data)
print(names == 'Bob') # [ True False False  True False False False]
print(data[names == 'Bob']) # 返回data的第0行和第3行，对应上面的True
print(data[names == 'Bob', 2:]) # 对于行，返回第0行和第3行；对于列，返回从第二列开始的元素

# 取反可以使用!=反转索引
print('Example 16:')
print(names != 'Bob') # [False  True  True False  True  True  True]
print(data[names != 'Bob'])

# 另一种取反方式是直接在表达式前使用~对条件取反
print('Example 17:')
print(data[~(names == 'Bob')]) 

# 使用｜(or)或者＆(and)进行布尔值条件联合
# python的关键字and和or在布尔值数组中没有用，需要用符号表示
print('Example 18:')
mask = (names == 'Bob') | (names == 'Will')
print(mask) # [ True False  True  True  True False False]
print(data[mask])

"""
神奇索引
注意神奇索引和切片的不同
切片是直接在一层中括号里写切片的索引
    arr[1:6]表示切1~6行
    arr[1:6, 2:5]表示切1~6行，2~5列
    切片只是取出地址，对于切片的所有操作都相当于在原数组上进行
神奇索引是在中括号中再用一层中括号写索引
    arr[[4, 6]]表示取出原数组第４行和第６行
    arr[[4, 6], [3, 5]]等价于取出行列索引为(4,3)和(6,5)的两个元素
    arr[[4, 6]][:, [3, 5]]表示先取第４行和第６行，再从取出的元素中取第3列和第5列
    神奇索引是对原数组的复制
"""
print('Example 19:')
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr) # [[0. 0. 0. 0.] [1. 1. 1. 1.] [2. 2. 2. 2.] [3. 3. 3. 3.] [4. 4. 4. 4.] [5. 5. 5. 5.] [6. 6. 6. 6.] [7. 7. 7. 7.]]

print('Example 20:')
print(arr[[4, 3, 0, 6]]) # [[4. 4. 4. 4.] [3. 3. 3. 3.] [0. 0. 0. 0.] [6. 6. 6. 6.]]
print(arr[[-3, -5, -7]]) # [[5. 5. 5. 5.] [3. 3. 3. 3.] [1. 1. 1. 1.]]

print('Example 21:')
arr = np.arange(32).reshape((8, 4)) 
print(arr) # [[ 0  1  2  3] [ 4  5  6  7] [ 8  9 10 11] [12 13 14 15] [16 17 18 19] [20 21 22 23] [24 25 26 27]　[28 29 30 31]]
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]]) # [ 4 23 29 10]
print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]) # [[ 4  7  5  6] [20 23 21 22] [28 31 29 30] [ 8 11  9 10]]

"""
数组转置和换轴
"""
# 转置
print('Example 22:')
arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.T)

# 转置＋计算内积
print('Example 23:')
arr = np.arange(15).reshape((3, 5))
print(arr)
print(np.dot(arr.T, arr))

# transpose()换轴
print('Example 24:')
arr = np.arange(16).reshape((2, 2, 4)) 
print(arr) # [[[ 0  1  2  3]  [ 4  5  6  7]] [[ 8  9 10 11]  [12 13 14 15]]]
print(arr.transpose((1, 0, 2))) # [[[ 0  1  2  3]  [ 8  9 10 11]] [[ 4  5  6  7]  [12 13 14 15]]]

