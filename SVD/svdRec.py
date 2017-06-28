# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:13:41 2017

@author: Qin
"""

import numpy as np

def loadExData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]
    
def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

# 相似度计算    
def euclidSim(inA, inB):
    '欧氏距离'
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))

def pearsSim(inA, inB):
    '皮尔逊相关系数'
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    '余弦相似度'
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num/denom)
    
# 基于物品相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
    '用来计算在给定相似度计算方法的条件下，用户对物品的估计评价值'
    # 获取数据集中物品的数目
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历数据集中的每个物品
    for j in range(n):
        # 获取已经被用户评分物品的分数
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 获取两个物品中已经被评分的那个元素
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item], dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal / simTotal
    
# 基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; 
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I 
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: 
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    '推荐引擎'
    # 对给定用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品则退出系统
    if len(unratedItems) == 0:
        return 'you rated everything'
    # 否则在每个未评分物品上进行循环
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append([item, estimatedScore])
    
    return sorted(itemScores, key=lambda t: t[1], reverse=True)[:N]

# 图像压缩函数
def printMat(inMat, thresh=0.8):
    '打印矩阵'
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end='')
            else: 
                print(0, end='')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    '图像压缩函数'
    myl = []
    for line in open('0_5.txt'):
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

if __name__ == '__main__':
    Data = np.mat(loadExData2())
    '''
    print(euclidSim(Data[:, 0], Data[:, 4]))
    print(euclidSim(Data[:, 0], Data[:, 0]))
    print('------------------------------------')
    print(pearsSim(Data[:, 0], Data[:, 4]))
    print(pearsSim(Data[:, 0], Data[:, 0]))
    print('------------------------------------')
    print(cosSim(Data[:, 0], Data[:, 4]))
    print(cosSim(Data[:, 0], Data[:, 0]))
    '''
    #print(recommend(Data, 2, estMethod=standEst))
    #print(recommend(Data, 2, estMethod=svdEst))
    print(imgCompress(2))