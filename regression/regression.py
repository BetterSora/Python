# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:31:34 2017

@author: Qin
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '数据导入函数'
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr:
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '标准回归函数'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    #linalg线性代数库
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return None
    ws = xTx.I * (xMat.T * yMat)
    
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    '局部加权线性回归函数'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    #创建对角矩阵
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return None
    ws = xTx.I * (xMat.T * (weights * yMat))
    
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    '为数据集中每个点调用lwlr'
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr -  yHatArr) ** 2).sum()

def ridgeRegress(xMat, yMat, lam = 0.2):
    '用于计算岭回归的回归系数'
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return None
    ws = denom.I * (xMat.T * yMat)
    
    return ws

def ridgeTest(xArr, yArr):
    '用于在一组lam上测试结果'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    #数据的标准化，求均值和方差
    yMean = np.mean(yMat)
    xMeans = np.mean(xMat, 0)
    #print(xMeans)
    #print(np.mean(xMat[0]))
    xVar = np.var(xMat, 0)
    #print(xVar)
    xMat = (xMat - xMeans) / xVar
    yMat = (yMat - yMean).T
    #在30个不同的lam下调用ridgeRegress()函数
    numTestPts = 30
    #用与存放30组ws
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    
    return wMat

def regularize(xMat):
    '按照均值0，方差1进行标准化处理(列方向)' 
    inMat = xMat.copy()  
    inMeans = np.mean(inMat,0) 
    inVar = np.var(inMat,0)
    inMat = (inMat - inMeans) / inVar
    
    return inMat  

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    '前向逐步回归'
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += sign * eps
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    
    return returnMat

if __name__ == '__main__':
    '''
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    #将点按升序排列
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, color = 'red')
    
    plt.show()
    
    print(lwlr(xArr[0], xArr, yArr, 1.0))
    print(lwlr(xArr[0], xArr, yArr, 0.001))
    '''
    abX, abY = loadDataSet('abalone.txt')
   
    wMat = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wMat)
    plt.show()
    
    #stageWise(abX, abY, 0.01, 5000)