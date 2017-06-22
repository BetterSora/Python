# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:35:07 2017

@author: Qin
"""

# FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    
    def inc(self, numOccur):
        '对count变量增加给定值'
        self.count += numOccur
        
    def disp(self, ind=1):
        '将树以文本形式显示'
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

# FP树构建函
def createTree(dataSet, minSup=1):
    # 项头表
    headerTable = {}
    # 给项头表添加数据
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for key in headerTable.copy():
        if headerTable[key] < minSup:
            del(headerTable[key])
    # 如果没有元素项满足要求则退出
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    # 对头指针表扩展以便可以保存计数值及指向每种类型第一个元素项的指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 创建只包含空集合的根节点
    retTree = treeNode('Null Set', 1, None)
    # 根据全局频率对每个事务中的元素进行排序
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # a = {'a':4, 'd':1, 'b':2}
            # sorted(a.items(), key=lambda p: p[1], reverse=True)
            # [('a', 4), ('b', 2), ('d', 1)]
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    
    return retTree, headerTable
          
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 已有该节点，count+1
        inTree.children[items[0]].inc(count)
    else:
        # 创建新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新项头表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    # 链表
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
        
    return retDict

# 发现以给定元素项结尾的所有路径函数
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    'treeNode:项头表指向的第一个节点 basePat:需要寻找条件模式基的节点'
    # 存储条件模式基
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            # 从1开始取的原因是条件模式基不包含叶节点
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
        
    return condPats

# 递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 从项头表的底端开始
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None: 
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    retTree, headerTable = createTree(initSet, 3)
    freqItems = []
    mineTree(retTree, headerTable, 3, set([]), freqItems)