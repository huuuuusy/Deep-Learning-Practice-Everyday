"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 使用kNN对手写数字进行识别
      trainingDigits包含约2000个例子
      testDigits包含约900个例子
      每个文本保存一个32*32的矩阵
"""
import numpy as np
import operator
from os import listdir
import collections

def img2vector(filename):
    """
    将32*32的二进制图像转化为1*1024的列表
    """
    vector = []
    fr = open(filename)
    fr_lines = fr.readlines()
    for line in fr_lines:
        line = list(line.strip('\n'))
        for num in line:
            vector.append(int(num))
    return vector

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

def handwritingClassTest():
    """
    手写数字分类测试
    """
    labels = []
    # 训练数据文件名
    trainingFileList = listdir('trainingDigits')
    # 返回训练数据文件夹下文件个数
    mTrain = len(trainingFileList)
    # 初始化训练矩阵
    trainingMat = np.zeros((mTrain, 1024))
    # 从文件名解析训练类别
    for i in range(mTrain):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        labels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 测试数据文件名
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        testVector = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = classify(testVector, trainingMat, labels, 3)
        print('classify result%d\tground truth%d' % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1
    print("total error nums:%d \n error rate:%f"%(errorCount, errorCount/mTest))


if __name__ == '__main__':
    handwritingClassTest()