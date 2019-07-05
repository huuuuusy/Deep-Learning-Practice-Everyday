"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
工具： python == 3.7.3
介绍： 使用KNN进行最简单的电影分类
      在V1的基础上优化代码
"""

import numpy as np 
import operator
import collections

def create_data():
    """
    生成程序需要的数据和标签
    """
    dataset = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['love movie', 'love movie', 'action movie', 'action movie']
    return dataset, labels

def classify(test, dataset, labels, k):
    """
    KNN算法分类器
    """
    # 计算test与dataset之间的距离
    distance = np.sum((test - dataset)**2, axis = 1)**0.5
    print('distance:\n', distance)
    # 统计k个最近的标签
    k_label_index = distance.argsort()[0:k]
    k_label = [labels[i] for i in k_label_index]
    print('top k labels:\n', k_label)
    # 输出最终结果
    label = collections.Counter(k_label).most_common(1)[0][0]
    return label

if __name__ == '__main__':
	# 创建数据集
	group, labels = create_data()
	# 测试集
	test = [101,20]
	# kNN分类
	test_class = classify(test, group, labels, 3)
	# 打印分类结果
	print('final result:\n', test_class)
        

