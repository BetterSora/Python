# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:14:53 2017

@author: Qin
"""

import numpy as np
import feedparser

def loadDataSet():
    '创建一些实验样本'
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字
    
    return postingList, classVec

def createVocabList(dataSet):
    '创建一个包含在所有文档中出现的不重复词的列表'
    vocaSet = set()
    for document in dataSet:
        vocaSet = vocaSet | set(document)
    
    return list(vocaSet)

def setOfWords2Vec(vocabList, inputSet):
    '词集模型'
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    '词袋模型'
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '朴素贝叶斯分类器训练函数'
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p0Denom = 2.0
    p1Num = np.ones(numWords)
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '朴素贝叶斯分类函数'
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    '便利函数'
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    '文本解析'
    import re
    '''
    Your regular expression ('\\W*') is invalid - it means zero or more not word characters. 
    Zero or more can be matched by "nothing", and if empty string is a match, where exactly do you expect splits will occur? 
    '''
    #listOfTokens = re.split(r'\W*', bigString)
    listOfTokens = re.split(r'\W+', bigString)
    
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '垃圾邮件测试函数'
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email\spam\%d.txt' % i, encoding = 'utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email\ham\%d.txt' % i, encoding = 'utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = [i for i in range(50)]
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector,p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    
def calcMostFreq(vocabList, fullText):
    '高频词去除函数'
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = lambda t : t[1], reverse = True)
    
    return sortedFreq[:30]
    
def localWords(feed1, feed0):
    'RSS源分类器'
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = [i for i in range(2 * minLen)]
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pFeed1 = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector,p0V, p1V, pFeed1) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    
    return vocabList, p0V, p1V
    
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -4.5:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -4.5:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key = lambda t : t[1], reverse = True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key = lambda t : t[1], reverse = True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])

if __name__ == '__main__':
    #testingNB()
    #spamTest()
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)