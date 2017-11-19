import operator
from math import log

def createDataSet():
	dataSet = [[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


def calcShannonEnt(dataSet):

	"""

		calcShannonEnt(dataSet) -> [shannonEnt]

		Paramter
			- dataSet 		: 数据集

		Return
			- shannonEnt 	: 信息熵

	"""

	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


def splitDataSet(dataSet, axis, value):

	"""

		splitDataSet() -> [retDataSet]

		Paramter
			- dataSet 		: 待划分的数据集
			- axis			: 划分数据集的特征
			- value			: 需要返回的特征的值

		Return
			- retDataSet	: 新的列表，划分后的数据集

	"""

	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):

	"""

		chooseBestFeatureToSplit() -> [bestFeature]
	
		Paramter
			- dataSet 		: 数据集

		Return
			- bestFeature 	: 最好的特征

	"""

	numFeatures = len(dataSet[0]) - 1
	baseEnntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEnntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


def majorityCnt(classList):

	"""

		majorityCnt() -> [sortedClassCount[0][0]]

		Paramter
			- classList					: 分类名称列表

		Return
			- sortedClassCount[0][0] 	: 出现次数最多的分类名称

	"""

	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), \
		key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):

	"""
	
		createTree() -> [myTree]
	
		Paramter
			- dataSet 		: 数据集
			- labels 		: 类别集
	
		Return
			- myTree		: 根节点

	"""

	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = \
			createTree(\
				splitDataSet(dataSet, bestFeat, value),\
				subLabels)
	return myTree


