from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):

	"""

		Classify0(inX, dataSet, labels, k) -> [sortedClassCount[0][0]]
	
		Parameter
			- inX 		: 用于分类的输入向量
			- dataSet 	: 输入的训练样本集
			- labels	: 标签向量
			- k			: 用于选择最近邻近的数目

		Return
			- sortedClassCount[0][0] :  最优结果

	"""

	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), 
		key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]


def file2matrix(filename):

	"""

		file2matrix(filename) -> [returnMat]，[classLabelVector] 
	
		Paramter
			- filename			: 文件名字

		Return
			- returnMat 		: 数据集矩阵
			- classLabelVector  : 数据集对应的分类矩阵

	"""

	fr = open(filename, 'r')
	arrayOLine = fr.readlines()
	numberOfLines = len(arrayOLine)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLine:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


def autoNorm(dataSet):

	"""
	
		autoNorm(dataSet) -> [normDataSet], [ranges], [minVals]

		Paramter
			- dataSet 			: 数据集

		Return
			- normDataSet 		: 数值归一化的数据集
			- ranges			: 最大值与最小值之间的插值（间距）
			- minVals			: 最小值集
	"""

	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals


def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
			datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real anser is: %d"\
			% (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]): errorCount += 1.0
	print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input(\
		"percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice-cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-\
		minVals)/ranges, normMat, datingLabels, 3)
	print("You will probably like this person: ",\
		resultList[classifierResult - 1])
