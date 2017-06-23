# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:50:09 2017

@author: Qin
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr]
    datArr = [list(map(float, line)) for line in stringArr]
    
    return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    # 求出数据每一维的均值
    meanVals = np.mean(dataMat, axis=0)
    # 去均值
    meanRemoved = dataMat - meanVals
    # 求协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 求协方差矩阵的特征值和特征向量(列向量)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 将特征值对应的下标按照特征值从小到大的顺序排列
    eigValInd = np.argsort(eigVals)
    # 获取最大的N个特征值的下标
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 获取特征值对应的特征向量
    redEigVects = eigVects[:,eigValInd]
    # 将数据变换到新的坐标域    
    lowDDataMat = meanRemoved * redEigVects
    # 对原始数据进行重构
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    
    return lowDDataMat, reconMat
    
if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDDataMat, reconMat = pca(dataMat, 1)
    plt.scatter(dataMat[:, 0], dataMat[:, 1], edgecolors='black', c='blue')
    plt.scatter(reconMat[:, 0], reconMat[:, 1], edgecolors='black', c='red')
    plt.show()