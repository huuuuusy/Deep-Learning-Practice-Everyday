"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.37
工具： python == 3.7.3
介绍： 使用sklearn库对kNN算法进行实现
      简单的手写数字识别，掌握 scikit-learn 机器学习库的基本使用。
      在这个例子中，图像像素的数值都在一定的范围内，归一化以后的效果不太明显，但这一步是必须的。
"""

"""
下载数据集
"""
from sklearn import datasets
from matplotlib import pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target 

# 查看数据集
print(X.shape) # (1797, 64)
print(X[0].shape) # (64,)
print(y.shape) # (1797,)
print(y[0]) # 0

plt.imshow(X[0,:].reshape((8,8)))
plt.show()

"""
数据预处理
"""
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

# 对数据进行归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

"""
构建模型，验证模型
"""
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5) # 构建模型，k取值为5
clf.fit(X_train_scaler,y_train) # 拟合数据，注意传入的是归一化后的训练数据
y_pred = clf.predict(X_test_scaler) # 测试，传入的也是归一化后的测试数据
score = clf.score(X_test_scaler, y_test) # 计算测试数据的得分
print(score) # 0.9722222222222222

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9722222222222222

"""
使用混淆矩阵看看分类效果
"""
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#[[38  0  0  0  0  0  0  0  0  0]
# [ 0 29  0  0  0  1  0  0  0  0]
# [ 0  0 34  0  0  0  0  0  1  0]
# [ 0  0  1 41  0  1  0  0  1  0]
# [ 0  0  0  0 42  0  0  0  0  0]
# [ 0  0  0  0  0 29  0  0  0  1]
# [ 0  0  0  0  0  0 30  0  0  0]
# [ 0  0  0  0  0  0  0 36  0  0]
# [ 0  0  0  0  0  1  0  0 38  0]
# [ 0  0  0  1  0  1  0  1  0 33]]
#　非零元素基本集中在对角线上，表示分类效果比较理想

"""
打印测试的评价指标
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#             precision    recall  f1-score   support
#           0       1.00      1.00      1.00        38
#           1       1.00      0.97      0.98        30
#           2       0.97      0.97      0.97        35
#           3       0.98      0.93      0.95        44
#           4       1.00      1.00      1.00        42
#           5       0.88      0.97      0.92        30
#           6       1.00      1.00      1.00        30
#           7       0.97      1.00      0.99        36
#           8       0.95      0.97      0.96        39
#           9       0.97      0.92      0.94        36
#    accuracy                           0.97       360
#   macro avg       0.97      0.97      0.97       360
#weighted avg       0.97      0.97      0.97       360