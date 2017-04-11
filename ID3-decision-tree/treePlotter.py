# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:39:33 2017

@author: Qin
"""

import matplotlib.pyplot as plt

#决策节点 boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度  
decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')
#叶节点
leafNode = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',
                            xytext = centerPt, textcoords = 'axes fraction',
                            va = 'center', ha = 'center', bbox = nodeType,
                            arrowprops = arrow_args)#xy 起点位置 xytext 注解框位置
                            
def plotMidText(cntrPt, parentPt, txtString):
    '在父子节点间填充文本信息'
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrpt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrpt, parentPt, nodeTxt)
    plotNode(firstStr, cntrpt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrpt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrpt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrpt, str(key))
            
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    '创建绘图区'
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    
def getNumLeafs(myTree):
    '获取叶节点的数目'
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '获取树的层数'
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]  
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
            
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    
    return listOfTrees[i]
    
if __name__ == '__main__':
    myTree = retrieveTree(0)
    createPlot(myTree)