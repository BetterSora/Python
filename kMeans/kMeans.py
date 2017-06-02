# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:40:10 2017

@author: Qin
"""

import numpy as np
import matplotlib.pyplot as plt

# kMeans支持函数
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr:
        curLine = line.strip().split('\t')
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    
    return dataMat

def distEclud(vecA, vecB):
    '距离计算函数'
    return np.sqrt(np.sum(np.power(vecA-vecB, 2)))

def randCent(dataSet, k):
    '产生k个随机质心'
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    # np.random.seed(10)
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        # np.random.rand(k, 1) 参数代表维度，范围始终是[0, 1)       
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    
    return centroids

# kMeans聚类算法
def kMeans(dataSet, k, distMeas=distEclud, creatCent=randCent):
    m = np.shape(dataSet)[0]
    # 聚类结果矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化质心
    centroids = creatCent(dataSet, k)
    # 用于标记簇是否改变
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(dataSet[i, :], centroids[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 只要有一个数据的簇发生改变，就改变标记
            if clusterAssment[i, :][0, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = [minIndex, minDist ** 2]
        print(centroids)
        # 更新质心位置
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0], :]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    
    return centroids, clusterAssment

# 二分kMeans
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    # 存储簇划分结果以及误差平方
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 初始化质心，将整个数据集看成一个簇
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 存储每个簇的质心
    centList = [centroid0]
    # 初始化误差平方
    for i in range(m):
        clusterAssment[i, 1] = distMeas(dataSet[i, :], np.mat(centroid0)) ** 2
    while len(centList) < k:
        lowestSSE = np.inf
        # 对当前数据集的每个簇进行遍历，看是否还需要进行划分
        for i in range(len(centList)):
            # 当前簇的数据集合
            ptsIncurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对当前簇进行划分
            centroidMat, splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)
            # 计算当前簇划分后的SSE值
            sseSplit = np.sum(splitClustAss[:, 1])
            # 计算未划分簇的SSE值
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1: ,].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:]= bestClustAss

    return np.mat(centList), clusterAssment

# 球面距离计算及簇绘图函数
def distSLC(vecA, vecB):
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * \
    np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
                      
    return np.arccos(a + b) * 6371.0 

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    # 前两个元素为axes的左下角在fig的图像坐标上的位置
    # 后两个元素指axes在fig的图像坐标上x方向和y方向的长度
    rect=[0.1,0.1,0.8,0.8]
    # 去除ax0的坐标轴坐标
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    # plt.xticks(())
    # plt.yticks(())
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    
    ax1.scatter(datMat[:, 0], datMat[:, 1], c=clustAssing[:, 0].A, s=90)
    ax1.scatter(myCentroids[:,0], myCentroids[:,1], marker='+', s=300)
  
    plt.show()

if __name__ == '__main__':
    '''
    dataMat = np.mat(loadDataSet('testSet2.txt'))
    centroids, clusterAssment = biKmeans(dataMat, 3)
    
    # 可视化
    plt.scatter(dataMat[:, 0], dataMat[:, 1], c=np.array(clusterAssment[:, 0]))
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=300, c='red')
    plt.show()
    '''
    clusterClubs()