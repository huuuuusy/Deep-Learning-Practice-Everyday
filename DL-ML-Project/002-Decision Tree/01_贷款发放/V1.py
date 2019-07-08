"""
@Author: huuuuusy
@GitHub: https://github.com/huuuuusy
系统： Ubuntu 18.04
IDE:  VS Code 1.36
工具： python == 3.7.3
介绍： 使用决策树判定是否需要对贷款申请者发放贷款
	  参考《机器学习实战》第三章，但是跳过决策树可视化部分
"""
from math import log
import operator
import pickle

def createDataSet():
    """
    创建贷款发放数据集
    """
    dataSet = [[0, 0, 0, 0, 'no'],
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels

def calcShannonEnt(dataSet):
	# 数据集大小
    numEntires = len(dataSet)    
	# 统计当前标签出现次数的字典                    
    labelCounts = {}                                
    for featVec in dataSet:                           
        currentLabel = featVec[-1]                    
        if currentLabel not in labelCounts.keys():   
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    
	# 计算香农熵            
    shannonEnt = 0.0                                
    for key in labelCounts:                           
        prob = float(labelCounts[key]) / numEntires    
        shannonEnt -= prob * log(prob, 2)           
    return shannonEnt           

def splitDataSet(dataSet, axis, value):
	"""
	按照给定特征划分数据集
	"""
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	选择最优特征,返回数据集信息增益最大的特征的索引值
	"""
	# 特征数量
	numFeatures = len(dataSet[0]) -1
	# 计算数据集香农熵
	baseEntropy = calcShannonEnt((dataSet))
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		# 获取dataSet的第i个所有特征
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		# 计算信息增益
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy -newEntropy
		# 存储信息增益最大的特征
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	"""
	统计classList中出现次数最多的元素（类标签）
	"""
	classCount = {}
	# 统计classList中每个元素出现的次数
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels, featLabels):
	"""
	创建决策树
	递归创建决策树时，递归有两个终止条件：
		1. 第一个停止条件是所有的类标签完全相同，则直接返回该类标签
		2. 第二个停止条件是使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用
		   此时说明数据纬度不够，由于第二个停止条件无法简单地返回唯一的类标签，这里挑选出现数量最多的类别作为返回值
	"""
	classList = [example[-1] for example in dataSet]
	# 如果类别完全相同，则停止划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 遍历完所有特征时返回出现次数最多的标签
	if len(dataSet[0]) == 1 or len(labels) == 0:
		return majorityCnt(classList)
	#选择最优特征
	bestFeat = chooseBestFeatureToSplit(dataSet)
	# 最优特征的标签
	bestFeatLabel = labels[bestFeat]							
	featLabels.append(bestFeatLabel)
	# 根据最优特征的标签生成树	
	myTree = {bestFeatLabel:{}}		
	# 删除已经使用特征标签			
	del(labels[bestFeat])			
	#得到训练集中所有最优特征的属性值							
	featValues = [example[bestFeat] for example in dataSet]		
	#去掉重复的属性值
	uniqueVals = set(featValues)								
	for value in uniqueVals:								
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
	return myTree

def classify(inputTree, featLabels, testVec):
	"""
	使用决策树进行分类
	"""
	# 获取决策树节点
	firstStr = next(iter(inputTree))
	# 下一个字典
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else: classLabel = secondDict[key]
	return classLabel

def storeTree(inputTree, filename):
	"""
	决策树存储
	"""
	with open(filename, 'wb') as fw:
		pickle.dump(inputTree, fw)

def grabTree(filename):
	"""
	决策树读取
	"""
	fr = open(filename, 'rb')
	return pickle.load(fr)

if __name__ == '__main__':
	dataSet, labels = createDataSet()
	featLabels = []
	# 保存决策树
	myTree = createTree(dataSet, labels, featLabels)
	storeTree(myTree, 'classifierStorage.txt')
	# 读取决策树
	myTree = grabTree('classifierStorage.txt')
	testVec = [0,1]
	result = classify(myTree, featLabels, testVec)
	if result == 'yes':
		print('放贷')
	if result == 'no':
		print('不放贷')