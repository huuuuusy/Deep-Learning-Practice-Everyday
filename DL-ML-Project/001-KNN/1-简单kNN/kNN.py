"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.35.1
工具： python == 3.7.3
介绍： 使用KNN进行最简单的电影分类
"""

import numpy as np 
import operator

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
    dataset_size = dataset.shape[0]
    # np.tile函数将test矩阵拓展成和dataset一样的形状
    # 即把test的行数复制成和dataset一致
    # diff计算拓展后的test矩阵和dataset的差值
    differents = np.tile(test, (dataset_size, 1)) - dataset
    square_differents = differents**2
    # 将差值的平方和按行相加
    square_distances = square_differents.sum(axis=1)
    # 开方后计算距离
    distances = square_distances**0.5
    # 按从小到大的顺序排序，并返回索引值
    sort_distances = distances.argsort()

    # 定义一个记录类别次数的字典
    class_count = {}

    for i in range(k):
        # 取出前k个元素的类别
        vote_label = labels[sort_distances[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值
		# 计算类别次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # key=operator.itemgetter(1)根据字典的值进行排序
	# key=operator.itemgetter(0)根据字典的键进行排序
	# reverse降序排序字典
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_class_count[0][0]

if __name__ == '__main__':
	# 创建数据集
	group, labels = create_data()
	# 测试集
	test = [101,20]
	# kNN分类
	test_class = classify(test, group, labels, 3)
	# 打印分类结果
	print(test_class)
        

