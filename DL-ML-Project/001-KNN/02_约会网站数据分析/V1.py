"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy

系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 使用kNN对datingTestSet进行分析，判定约会对象的所属类别
"""

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt 
import numpy as np 
import operator
import collections

def file2matrix(filename):
    """
    打开并解析文件，对数据进行分类：
    1代表不喜欢
    2代表魅力一般
    3代表极具魅力
    """
    fr = open(filename, 'r', encoding = 'utf-8')
    fr_lines = fr.readlines()
    # 针对有BOM的UTF-8文本，应去掉头部的BOM
    fr_lines[0] = fr_lines[0].lstrip('\ufeff')
    num_lines = len(fr_lines)
    # 解析完的数据应该有num_lines行,3列
    mat = np.zeros((num_lines, 3))
    # 标签向量
    labels = []
    # 行的索引
    index = 0

    for line in fr_lines:
        # 默认删除每行前后的空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 将list按照\t进行划分
        line_list = line.split('\t')
        # 将划分后的结果存入mat矩阵
        mat[index,:] = line_list[0:3]
        # 将文本最后一列的标签存入labels
        if line_list[-1] == 'didntLike':
            labels.append(1)
        elif line_list[-1] == 'smallDoses':
            labels.append(2)
        elif line_list[-1] == 'largeDoses':
            labels.append(3)
        index += 1
    return mat, labels

def autoNorm(dataset):
    """
    对数据进行归一化，确保所有特征取值范围在0~1
    """
    # 对每列数据取最大最小值
    # 0表示对列进行操作，1表示对行进行操作
    # 因为对列进行操作，mat矩阵一共3列，所以min_num返回3个数字,分别代表每个特征的最小值
    # max_num同理
    min_num = dataset.min(0)
    max_num = dataset.max(0)
    num_range = max_num - min_num
    norm_mat = (dataset - min_num) / num_range
    return norm_mat


def classify(test, dataset, labels, k):
    """
    KNN算法分类器
    """
    # 计算test与dataset之间的距离
    distance = np.sum((test - dataset)**2, axis = 1)**0.5
    # 统计k个最近的标签
    k_label_index = distance.argsort()[0:k]
    k_label = [labels[i] for i in k_label_index]
    # 输出最终结果
    label = collections.Counter(k_label).most_common(1)[0][0]
    return label

def datingClassTest():
    """
    分类器效果的测试函数
    """
    filename = 'datingTestSet.txt'
    mat, labels = file2matrix(filename)
    # 测试数据所占比例
    test_ratio = 0.1
    # 数据归一化
    norm_mat = autoNorm(mat)
    # 测试数据的个数
    num_test = int(test_ratio * norm_mat.shape[0])
    # 分类错误的个数
    num_error = 0.0

    for i in range(num_test):
        # 取前num_test个数据作为测试集，后面的数据作为dataset
        result = classify(mat[i,:], mat[num_test:,:], labels[num_test:],3)
        print("test result: %s \t ground truth: %s" % (result, labels[i]))
        if result != labels[i]:
            num_error += 1
    print("error ratio: %f" % (num_error/float(num_test)))


if __name__ == '__main__':
    datingClassTest()
    
        


