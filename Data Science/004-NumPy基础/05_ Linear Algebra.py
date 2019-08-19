"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用NumPy进行线性代数计算，参考《利用Python进行数据分析》4.5
"""
import numpy as np

"""
矩阵的点积
NumPy中*表示矩阵元素的逐元素乘积
对于矩阵的点积，需要使用np.dot()函数进行操作
"""
print('Example 1:')
x = np.array([[1., 2., 3.], [4., 5., 6.]]) # x是2*3矩阵
y = np.array([[6., 23.], [-1, 7], [8, 9]]) # y是3*2矩阵
print(x.dot(y)) # 点乘的结果是2*2矩阵　[[ 28.  64.] [ 67. 181.]]
print(np.dot(x, y)) # x.dot(y)等价于np.dot(x,y) [[ 28.  64.] [ 67. 181.]]

print('Example 2:')
print(np.dot(x, np.ones(3))) # [ 6. 15.]
print(x @ np.ones(3)) # 符号@作为中缀操作符，用于矩阵点积的操作[ 6. 15.]

"""
numpy.linalg函数集
"""
from  numpy.linalg import inv, qr
print('Example 3:')
X = np.array([[6., 23.], [-1, 7], [8, 9]])
mat = X.T.dot(X) #  X.T.dot(X)计算的是X和其转置矩阵X.T的点积　
print(mat) # [[101. 203.] [203. 659.]]
print(inv(mat)) # inv()计算的是方阵的逆矩阵　[[ 0.02599606 -0.00800789] [-0.00800789  0.00398422]]
print(mat.dot(inv(mat))) # [[1.00000000e+00 1.11022302e-16] [0.00000000e+00 1.00000000e+00]]
q, r = qr(mat) # qr()计算的是QR分解结果
print(q) # [[-0.44544857 -0.89530753] [-0.89530753  0.44544857]]
print(r) # [[-226.73773396 -680.43372097] [   0.          111.8031814 ]]
