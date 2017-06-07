# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:35:35 2017

@author: Qin
"""

# Apriori算法中的辅助函数
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    'C1是大小为1的所有候选项集的集合'
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    
    return [frozenset(i) for i in C1]

def scanD(D, Ck, minSupport):
    '参数分别为：数据集，候选项集列表，最小支持度'
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                try:
                    ssCnt[can] += 1
                except KeyError:
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 过滤集合
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    
    return retList, supportData

# Apriori算法
def aprioriGen(Lk, k):
    '生成Ck'
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j])
    
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = [set(i) for i in dataSet]
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):
    '参数：频繁项集列表，包含那些频繁项集支持数据的字典，最小可信度阈值'
    # 包含可信度的规则列表
    bigRuleList = []
    # 因为无法从单元素项集中构建关联规则，所以下标从1开始
    for i in range(1, len(L)):
        # freqSet是每个频繁项集
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 保存满足最小可信度的规则
    prunedH = [] 
    for conseq in H:
        # frozenset([3,4])-frozenset([3]) = frozenset({4})
        conf = supportData[freqSet] / supportData[freqSet - conseq] 
        if conf >= minConf: 
            print(freqSet - conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7, m=1):
    if len(freqSet) > m: 
        if m == 1:
            Hmp1 = H.copy()
        else:
            Hmp1 = aprioriGen(H, m)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):   
            m += 1
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf, m)

if __name__ == '__main__':
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    generateRules(L, supportData, minConf=0.7)