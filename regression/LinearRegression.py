# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:29:22 2017

@author: Qin
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def loadData():
    'a function to load data'
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=10)
    
    return X, y

def gradientDescent(X, y, theta, alpha, iterNum):
    '梯度下降算法'
    # 同时迭代更新参数
    theta = theta.copy()
    m = len(X)
    n = len(X[0])
    y_axis = []
    x_axis = []

    for i in range(iterNum):
        for j in range(n):
            theta[j] = theta[j] - alpha * ((np.dot(X, theta) - y) * X[:, j]).sum() / m       
        # cost function
        cost = np.power(np.dot(X, theta) - y, 2).sum() / (2 * m)
        y_axis.append(cost)
        x_axis.append(i)
    
    plt.plot(x_axis, y_axis)
    plt.show()
    
    return theta

def gradientDescentWithRegulation(X, y, theta, alpha, lam, iterNum):
    '梯度下降算法(正则化)'
    # 同时迭代更新参数
    theta = theta.copy()
    m = len(X)
    n = len(X[0])
    y_axis = []
    x_axis = []

    for i in range(iterNum):
        for j in range(n):
            if j == 0:
                theta[j] = theta[j] - alpha * ((np.dot(X, theta) - y) * X[:, j]).sum() / m   
            else:
                theta[j] = (1 - alpha * lam / m) * theta[j] - alpha * ((np.dot(X, theta) - y) * X[:, j]).sum() / m
        # cost function
        cost = (np.power(np.dot(X, theta) - y, 2).sum() + lam * np.power(theta[1:], 2).sum()) / (2 * m)
        y_axis.append(cost)
        x_axis.append(i)
    
    plt.plot(x_axis, y_axis)
    plt.show()
    
    return theta

def linearRegression1(X, y):
    '线性回归(梯度下降法)'
    # 初始化参数
    theta = [1] * len(X[0])
    alpha = 0.01
    iterNum = 1000
    lam = 1
    
    #theta = gradientDescent(X, y, theta, alpha, iterNum)
    theta = gradientDescentWithRegulation(X, y, theta, alpha, lam, iterNum)
    
    return theta
    
def linearRegression2(X, y):
    '线性回归(标准方程法)'
    X = np.matrix(X)
    y = np.matrix(y).T
    n = len(X[0]) + 1
    lam = 1

    #theta = (X.T * X).I * X.T * y
    
    # 正则化(还能避免矩阵出现不可逆的情况)
    reg = np.eye(n)
    reg[0, 0] = 0
    print(reg)
    theta = (X.T * X + lam * reg).I * X.T * y
    
    return theta
    
if __name__ == '__main__':
    X, y = loadData()   
    
    # 准备工作
    trainingData = np.ones(len(X))
    # 给数组增加一列
    trainingData = np.column_stack((trainingData, X))
    labels = y
    
    theta1 = linearRegression1(trainingData, y)
    print('theta1', theta1)
    theta2 = linearRegression2(trainingData, y)
    print('theta2', theta2)
    
    yHat = np.dot(trainingData, theta1)
    plt.scatter(X, y, edgecolors='black')
    plt.plot(X, yHat, color='red')
    plt.show()