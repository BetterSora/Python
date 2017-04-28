# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:58:48 2017

@author: Qin
"""

import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = np.mat([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '通过阈值比较对数据进行分类,dimen是选择哪个属性进行分类'
    retArray = np.ones((np.shape(dataMatrix)[0], 1))#初始化类别为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
        
    return retArray
        
def buildStump(dataArr, classLabels, D):
    '找到数据集上的最佳单层决策树'
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    #这个字典用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    #第一层for循环在数据集的所有特征上遍历
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            #在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                #计算加权错误率
                weightedError = D.T * errArr
                #print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    '基于单层决策树的AdaBoost训练过程'
    #对弱分类器进行存储
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    #记录类别累计
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #print('D:', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print('aggClassEst', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print('total error:', errorRate)
        if errorRate == 0.0:
            break
        
    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    'AdaBoost分类函数'
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    
    return np.sign(aggClassEst)

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr:
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
        
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    #绘制光标的值
    cur = (1.0, 1.0)
    #计算AUC的值
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    #获得排序的索引，必须用于行向量
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep 
            delY = 0
            #记录高度
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
        cur = (cur[0] - delX, cur[1] - delY)
    #随机猜测的结果曲线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is: ', ySum * xStep)

if __name__ == '__main__':
    #dataMat, classLabels = loadSimpData()
    #D = np.mat(np.ones((5, 1)) / 5)
    #bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    #weakClassArr = adaBoostTrainDS(dataMat, classLabels)
    #print(adaClassify([0, 0], weakClassArr))
    dataMat, classLabels = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataMat, classLabels, 50)
    #testArr, testClassLabels = loadDataSet('horseColicTest2.txt')
    #prediction10 = adaClassify(testArr, weakClassArr)
    #errArr = np.mat(np.ones((67, 1)))
    #errArr[prediction10 == np.mat(testClassLabels).T] = 0
    #print(errArr.sum())
    plotROC(aggClassEst.T, classLabels)