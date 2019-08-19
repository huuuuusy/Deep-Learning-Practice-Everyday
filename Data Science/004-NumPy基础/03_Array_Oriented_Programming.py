"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用NumPy数组进行面向数组编程，参考《利用Python进行数据分析》4.3
"""
import numpy as np

"""
向量化
"""
print('Example 1:')
points = np.arange(-5, 5, 0.01)
xs, ys  = np.meshgrid(points, points) # meshgrid()接收两个一维数组，并将两个数组中所有(x,y)生成一个二维矩阵
print(ys) # [[-5.   -5.   -5.   ... -5.   -5.   -5.  ] [-4.99 -4.99 -4.99 ... -4.99 -4.99 -4.99] [-4.98 -4.98 -4.98 ... -4.98 -4.98 -4.98] ... [ 4.97  4.97  4.97 ...  4.97  4.97  4.97] [ 4.98  4.98  4.98 ...  4.98  4.98  4.98] [ 4.99  4.99  4.99 ...  4.99  4.99  4.99]]

print('Example 2:')
z = np.sqrt(xs**2 + ys**2) 
print(z) # [[7.07106781 7.06400028 7.05693985 ... 7.04988652 7.05693985 7.06400028] [7.06400028 7.05692568 7.04985815 ... 7.04279774 7.04985815 7.05692568] [7.05693985 7.04985815 7.04278354 ... 7.03571603 7.04278354 7.04985815] ... [7.04988652 7.04279774 7.03571603 ... 7.0286414  7.03571603 7.04279774] [7.05693985 7.04985815 7.04278354 ... 7.03571603 7.04278354 7.04985815] [7.06400028 7.05692568 7.04985815 ... 7.04279774 7.04985815 7.05692568]]

print('Example 3:')
import matplotlib.pyplot as plt
plt.imshow(z)
plt.draw()
# plt.savefig("03_01.png")

"""
将条件逻辑作为数组操作
"""
print('Example 4:')
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)] # 直接使用列表进行操作，代码复杂，生成的结果是list形式
print(result) # [1.1, 2.2, 1.3, 1.4, 2.5]

"""
where()函数
"""
print('Example 5:')
result = np.where(cond, xarr, yarr) # 使用np.where函数，代码简洁，生成的是ndarray
print(result) # [1.1 2.2 1.3 1.4 2.5]

print('Example 6:')
arr = np.random.randn(2, 2) # 随机生成矩阵数据
print(arr) # 2*2的随机数矩阵 [[ 1.74934151 -1.54214403] [-1.12455213 -0.48854961]]
print(arr > 0) # 逐元素和0比较　[[ True False] [False False]]
print(np.where(arr > 0, 2, -2)) #　逐元素替换 [[ 2 -2] [-2 -2]]
print(np.where(arr > 0, 2, arr)) # [[ 2.         -1.54214403] [-1.12455213 -0.48854961]]

"""
数学运算
"""
print('Example 7:')
arr = np.random.randn(2, 3)
print(arr) # [[-1.10311774 -1.36369792  0.34821223] [ 0.56715181 -0.0860741  -2.03733694]]
print(arr.mean()) # -0.6124771112647218
print(np.mean(arr)) # -0.6124771112647218
print(arr.sum()) # -3.674862667588331
print(arr.mean(axis=1)) # axis = 1表示沿着横轴运算　[-0.70620115 -0.51875308]
print(arr.sum(axis=0)) # axis = 0表示沿着纵轴运算　[-0.53596593 -1.44977203 -1.68912471]

print('Example 8:')
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr) # [[0 1 2] [3 4 5] [6 7 8]]
print(arr.cumsum(axis=0)) # cumsum表示从0开始的累计和　[[ 0  1  2] [ 3  5  7] [ 9 12 15]]
print(arr.cumprod(axis=1)) # cumprod表示从１开始的累计积[[  0   0   0] [  3  12  60] [  6  42 336]]

"""
布尔值数组
"""
print('Example 9:')
arr = np.random.randn(100)
print((arr > 0).sum()) # 统计随机产生的ndarray中正数的个数　
bools = np.array([False, False, True, False])
print(bools.any()) # any()检查数组中是否至少有一个True 
print(bools.all()) # all()检查数组是否均为True

"""
排序
"""
print('Example 10:')
arr  = np.random.randn(6)
print(arr) # [-2.31933477 -0.05466234 -0.80974333  0.35521019  0.74005554 -0.02727645]
arr.sort() #　sort()按位置排序
print(arr) # [-2.31933477 -0.80974333 -0.05466234 -0.02727645  0.35521019  0.74005554]

print('Example 11:')
arr = np.random.randn(2,3)
print(arr) # [[-0.35574883 -0.07969707  1.70822511] [-0.75730269 -0.82255508  0.3292006 ]]
arr.sort(1) # sort()可以指定轴向排序，0为纵轴，1为横轴
print(arr) # [[-0.35574883 -0.07969707  1.70822511] [-0.82255508 -0.75730269  0.3292006 ]]

"""
唯一值
np.unique()返回的是数组中唯一值排序后形成的数组
相当于sorted(set(arr))
"""
print('Example 12:')
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names)) # ['Bob' 'Joe' 'Will']
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
print(np.unique(ints)) # [1 2 3 4]

"""
集合逻辑
np.in1d()检查一个数组中的值是否在另一个数组中
"""
print('Example 13:')
values = np.array([6, 0, 0, 3, 2, 5, 6])
print(np.in1d(values, [2, 3, 6])) # [ True False False  True  True False  True]

