# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:47:11 2017

@author: Qin
"""

import numpy as np

def loadDataSet():
    '便利函数'
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        
    return dataMat, labelMat

def sigmoid(inX):
    'sigmoid函数'
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '梯度上升法'
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    
    return weights

def stocGradAscent0(dataMatIn, classLabels):
    '随机梯度上升法'
    dataArr = np.array(dataMatIn)
    m, n = np.shape(dataArr)
    alpha = 0.01
    weights = np.ones(n)
    #for j in range(200):
    for i in range(m):
        h = sigmoid(sum(dataArr[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * dataArr[i] * error
    
    return weights

def stocGradAscent1(dataMatIn, classLabels, numIter = 150):
    '改进的随机梯度上升法'
    dataArr = np.array(dataMatIn)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = [i for i in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataArr[randIndex] * error
            del(dataIndex[randIndex])
    
    return weights

def stocGradAscent2(dataMatIn, classLabels, numIter = 150):
    '自己尝试的随机梯度上升法'
    dataArr = np.array(dataMatIn)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = [i for i in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataArr[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * dataArr[dataIndex[randIndex]] * error
            del(dataIndex[randIndex])
    
    return weights

def stocGradAscent3(dataMatIn, classLabels, numIter = 150):
    '自己尝试的随机梯度上升法'
    dataArr = np.array(dataMatIn)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            h = sigmoid(sum(dataArr[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * dataArr[i] * error
    
    return weights

def plotBestFit(weights):
    '画出数据集和Logistic回归最佳拟合直线的函数'    
    import matplotlib.pyplot as plt
    weights = np.array(weights)
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    m = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []#类别1的坐标
    xcord2 = []; ycord2 = []#类别0的坐标
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

#Logistic分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain:
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0.0; numTestVec = 0.0
    for line in frTest:
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classifyVector(np.array(lineArr), trainWeights) != float(currLine[21]):
            errorCount += 1
    errorRate = errorCount / numTestVec
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    #dataArr, labelMat = loadDataSet()
    #weights = gradAscent(dataArr, labelMat)
    #plotBestFit(weights)
    #weights = stocGradAscent0(dataArr, labelMat)
    #plotBestFit(weights)
    #weights = stocGradAscent1(dataArr, labelMat)
    #print(weights)
    #plotBestFit(weights)
    #weights = stocGradAscent2(dataArr, labelMat)
    #plotBestFit(weights)
    #weights = stocGradAscent3(dataArr, labelMat)
    #plotBestFit(weights)
    multiTest()