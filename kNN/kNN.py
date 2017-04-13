# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:54:20 2017

@author: Qin
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    
    return (group, labels)

def autoNorm(dataSet):
    '归一化特征值'
    minVals = dataSet.min(0)#list
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    
    return (normDataSet, ranges, minVals)

def classify0(inX, dataSet, labels, k):
    'k-近邻算法'
    #距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)#行求和
    distances = sqDistances ** 0.5
    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()#按从小到大将对应的坐标进行排列
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #assert isinstance(voteIlabel, int)
        try:
            classCount[voteIlabel] += 1
        except KeyError:
            classCount[voteIlabel] = 1
    sortedClassCount = sorted(classCount.items(), key = lambda t : t[1], reverse = True)
    
    return sortedClassCount[0][0]

def file2matrix(filename):
    f = open(filename)
    arrayOLines = f.readlines()
    f.close()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVecter = []
    index = 0
    
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0 : 3]
        classLabelVecter.append(int(listFromLine[-1]))
        index += 1
        
    return (returnMat, classLabelVecter)

def datingClassTest():
    hoRatio = 0.1
    (datingDataMat, datingLabels) = file2matrix('datingTestSet2.txt')
    (normMat, ranges, minVals) = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('the total error rate is: %f' %(errorCount / numTestVecs))
    print(errorCount)
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent in playing vedio games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    (datingDataMat, datingLabels) = file2matrix('datingTestSet2.txt')
    (normMat, ranges, minVals) = autoNorm(datingDataMat)
    inArr = np.array([percentTats, ffMiles, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print('You will probably like this person: %s' %(resultList[classifierResult - 1]))
    
'''---------------------------------------------------------------------------------------------------------------------------'''

def img2vector(filename):
    '将图像转换为测试向量'
    returnVect = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = lineStr[j]
    f.close()
    
    return returnVect

def handwritingClassTest():
    '手写数字识别系统的测试代码'
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' %(fileNameStr))
    
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('the total error rate is: %f' %(errorCount / float(mTest)))
    print('the total number of errors is: %d' %(errorCount))

if __name__ == '__main__':
    #(datingDataMat, datingLabels) = file2matrix('datingTestSet2.txt')
    #fig = plt.figure()
    #ax = fig.add_subplot('111')
    #ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
    #           15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    #plt.show()
    
    #(normMat, ranges, minVals) = autoNorm(datingDataMat)
    
    #datingClassTest()
    #classifyPerson()
    
    handwritingClassTest()