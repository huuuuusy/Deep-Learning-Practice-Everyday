"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用sklearn库对kNN算法进行实现
      预测鸢尾花数据集（官网例子）
      鸢尾花数据集包含花的多种属性，取前两个属性进行分类
      鸢尾花可以被分为三类，y的取值分别为0,1,2
      http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


"""
下载数据集
"""

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

print(X[:5])
#[[5.1 3.5]
# [4.9 3. ]
# [4.7 3.2]
# [4.6 3.1]
# [5.  3.6]]

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# 2 2]

"""
设置参数
"""
# 每一个网格之间的间距，在后面的 np.meshgrid 方法中可以用到
h = 0.02 

# kNN的k值
n_neighbors = 15

# ListedColormap 传入一个颜色列表，可以放多一些，因为如果颜色不够用的话，就会循环选取，会影响可视化的结果。
# 背景网格的颜色，应该选择浅一些的颜色
cmap_light = ListedColormap(['#FFFFF0','#B0E2FF','#FFE1FF'])
# 训练数据集的颜色，应该选择深一些的颜色
cmap_bold = ListedColormap(['r','b','g'])

"""
绘图：使用带有权重的kNN进行决策，权重分别取uniform和distance
"""
for weights in ['uniform','distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights)
    clf.fit(X, y)
    
    # 画出决策边界
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max() + 1
    # 画出网格
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                          np.arange(x2_min, x2_max, h))
    # 对于上面的每一个点，都使用 kNN 算法预测一下它们的类别
    Z = clf.predict(np.c_[xx1.ravel(),xx2.ravel()])
    # 这里得到的 Z 的形状是拉平了以后的，
    # 要在 matplotlib 中画出图，需要 reshape 成网格的形状
    # 即 xx1 的形状，或者 xx2 的形状
    # 或者 Z = Z.reshape(xx2.shape)
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(8,6))
    
    # 画出网格，使用亮色，因为这些密密麻麻的点都是背景
    plt.pcolormesh(xx1,xx2,Z,cmap = cmap_light)
    # 画出训练数据集那些点
    # x 表示直角坐标系中 x 轴的坐标值
    # y 表示直角坐标系中 y 轴的坐标值
    # c 表示 scatter 出来的那些点的颜色，会根据 cmap 参数提供的 ListedColormap 对象中的列表依次选取
    # edgecolor 表示 scatter 出来的那些点的边的颜色
    # s 的意思是 size，scatter 出来的那些点的大小
    plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title('3 - 分类问题（k = {}，weights = {}）'.format(n_neighbors,weights),fontsize=18)
    plt.savefig('3 - 分类问题（k = {}，weights = {}）'.format(n_neighbors,weights) +'.png')
    plt.show()