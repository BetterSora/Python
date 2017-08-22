# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:54:45 2017

@author: Qin
"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing

def loadData(file, delimiter):
    '加载数据'
    data = np.loadtxt(file, delimiter=delimiter)
    print('Dimension: ', data.shape)
    print(data[:6, :])
    
    return data

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    '画出数据点'
    pos = data[:, 2] == 1
    neg = data[:, 2] == 0
    
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], marker='o', c='y', s=60, edgecolor='k', label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)

def sigmoid(z):
    'sigmoid函数'
    s = 1 / (1 + np.exp(-z))
    
    return s

def costFunction(theta, X, y):
    'cost function'
    m = len(X)
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    return J[0]

def costFunction2(theta, lam, X, y):
    'cost function(正则化)'
    m = len(X)
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + lam / (2 * m) * np.power(theta[1:], 2).sum()

    return J[0]
 
def gradientDescent(X, y, theta, alpha, iterNum):
    '梯度下降法'
    m = len(X)
    theta = theta.copy()
    x_axis = []
    y_axis = []
    for i in range(iterNum):
        theta = theta - alpha / m * (X.T.dot((sigmoid(X.dot(theta)) - y)))
        J = costFunction(theta, X, y)
        y_axis.append(J)
        x_axis.append(i)
        if i % 100 == 0:
            print('cost: ', J)
    #axes = plt.gca()
    #axes.plot(x_axis, y_axis)
    
    return theta

def gradientDescent2(X, y, theta, alpha, iterNum):
    '梯度下降法(带正则化项)'
    m = len(X)
    lam = 1
    theta = theta.copy()
    x_axis = []
    y_axis = []
    for i in range(iterNum):
        theta[0] = theta[0] - alpha / m * (X.T[0].dot((sigmoid(X.dot(theta)) - y)))
        theta[1:] = theta[1:] * (1 - alpha * lam / m) - alpha / m * (X.T[1:].dot((sigmoid(X.dot(theta)) - y)))
        J = costFunction2(theta, lam, X, y)
        y_axis.append(J)
        x_axis.append(i)
        if i % 100 == 0:
            print('cost: ', J)
    #axes = plt.gca()
    #axes.plot(x_axis, y_axis)
    
    return theta

def logisticRegression(X, y):
    'logistic回归'
    # 初始化参数
    theta = np.zeros((X.shape[1], 1)) 
    alpha = 0.3
    iterNum = 3000
    
    #theta = gradientDescent(X, y, theta, alpha, iterNum)
    theta = gradientDescent2(X, y, theta, alpha, iterNum)
    
    return theta

if __name__ == '__main__':
    '''
    data = loadData('data1.txt', ',')
    std = np.std(data[:, :2], axis=0)
    mean = np.mean(data[:, :2], axis=0)
    data[:, :2] = (data[:, :2] - mean) / std
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    #data[:, :2] = min_max_scaler.fit_transform(data[:, :2])

    X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
    y = np.c_[data[:, 2]]
    clf = linear_model.LogisticRegression(max_iter=100)
    clf.fit(data[:, 0:2], data[:, 2])
    print('sklearn logistic: ', clf.intercept_, clf.coef_) 
    
    theta = logisticRegression(X, y)
    print('my logistic: ', theta.flatten())
    
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
    x1_min, x1_max = data[:, 0].min(), data[:, 0].max()
    x2_min, x2_max = data[:, 1].min(), data[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, 1, linewidths=1, colors='b');
    '''
    # 多项式logistic回归
    data2 = loadData('data2.txt', ',')
    poly = preprocessing.PolynomialFeatures(6)
    XX = poly.fit_transform(data2[:, 0:2])
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #XX = min_max_scaler.fit_transform(XX)
    
    X = np.c_[np.ones((len(XX), 1)), XX]
    y = np.c_[data2[:, 2]]
    
    theta = logisticRegression(X, y)
    print('my logistic: ', theta.flatten())
    
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
    x1_min, x1_max = data2[:,0].min(), data2[:,0].max()
    x2_min, x2_max = data2[:,1].min(), data2[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((len(xx1.ravel()), 1)), poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()])].dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, 1, linewidths=1, colors='g');       