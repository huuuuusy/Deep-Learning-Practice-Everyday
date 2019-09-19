"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用sklearn库对kNN算法进行实现
      入门小例子
"""

"""
使用所有训练数据构成特征 X 和标签 y，使用 fit() 函数进行训练。
在正式分类时，通过一次性构造测试集或者一个一个输入样本的方式，得到样本对应的分类结果。
有关 K 的取值:
    如果较大，相当于使用较大邻域中的训练实例进行预测，可以减小估计误差，但是距离较远的样本也会对预测起作用，导致预测错误。
      【k 较大的时候，模型越简单，比较容易欠拟合。】
    相反地，如果 K 较小，相当于使用较小的邻域进行预测，如果邻居恰好是噪声点，会导致过拟合。
      【k 较小的时候，模型越复杂，比较容易过拟合。】 
    一般情况下，K 会倾向选取较小的值，并使用交叉验证法选取最优 K 值。
"""

from sklearn.neighbors import KNeighborsClassifier

# 原始数据集
X = [[0],[1],[2],[3]]
y = [0,0,1,1]

clf = KNeighborsClassifier(n_neighbors=3) # 设定k值为3
clf.fit(X,y) # 拟合数据
print(clf.predict([[1.1]])) # 对新样本[1.1]进行预测，预测的分类结果是[0]