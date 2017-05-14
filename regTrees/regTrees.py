# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:02:11 2017

@author: Qin
"""

import numpy as np

def loadDataSet(filename):
    '数据加载函数'
    dataMat = []
    fr = open(filename)
    for line in fr:
        curLine = line.strip().split('\t')
        #将每行映射成浮点数
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
        
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    '在给定特征和特征值的情况下，切分数据集'
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    
    return mat0, mat1

def regLeaf(dataSet):
    '创建叶节点'
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    '计算总方差'
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '数据集切分的最佳位置'
    tolS = ops[0]
    tolN = ops[1]
    #如果所有值相等则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #如果切分出的数据集很小则退出
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '树创建函数'
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    
    return retTree

#----------------------------------后剪枝---------------------------------------
def isTree(obj):
    '判断是否是一棵子树'
    return isinstance(obj, dict)

def getMean(tree):
    '对树进行塌陷处理(即返回树平均值)'
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0

def prune(tree, testData):
    '后剪枝'
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if (not isTree(tree['right'])) and (not isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

#----------------------------------模型树---------------------------------------
def linearSolve(dataSet):
    '将数据集格式化成目标变量Y和自变量X'
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1 : n] = dataSet[:, 0 : n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    
    return ws, X, Y

def modelLeaf(dataSet):
    '生成叶节点'
    ws, X, Y = linearSolve(dataSet)
    
    return ws

def modelErr(dataSet):
    '在给定数据集上计算误差'
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    
    return sum(np.power(yHat - Y, 2))

#-----------------------------用树回归进行预测-----------------------------------
def regTreeEval(model, inDat):
    '回归树叶节点预测值'
    return float(model)

def modelTreeEval(model, inDat):
    '模型树叶节点预测值'
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    #对输入数据进行格式化处理，并加入第0列
    X[:, 1 : n + 1] = inDat
    
    return float(X * model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
    '自顶向下遍历整棵树，知道遇到叶节点为止'
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[0, tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
        
    return yHat

if __name__ == '__main__':
    #创建一棵回归树
    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree1 = createTree(trainMat, ops = (1, 20))
    yHat1 = createForeCast(myTree1, testMat)
    print(np.corrcoef(yHat1, testMat[:, 1], rowvar = 0)[0, 1])
    #创建一棵模型树
    myTree2 = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    print(np.corrcoef(yHat2, testMat[:, 1], rowvar = 0)[0, 1])
    #标准线性回归
    ws, X, Y = linearSolve(trainMat)
    m = np.shape(testMat)[0]
    yHat3 = np.zeros((m, 1))
    for i in range(np.shape(testMat)[0]):
        yHat3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(np.corrcoef(yHat3, testMat[:, 1], rowvar = 0)[0, 1])