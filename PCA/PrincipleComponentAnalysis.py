# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:55:38 2017

@author: Qin
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    # 加载数据
    mat = sio.loadmat(filename)
    X = mat['X']
    
    return X

def featureNormalize(X):
    # normalize
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    
    return X_norm

def pca(X):
    # principle component analysis
    m = len(X)
    cov_mat = np.dot(X.T, X) / m
    U, S, V = np.linalg.svd(cov_mat, full_matrices=True)
    '''
    # 降维
    K = 1
    Z = np.dot(X, U[:, :K]) # m*k
    print(Z[0, 0])

    # 恢复
    X_rec = np.dot(Z, U[:, :K].T)
    print(X_rec[0, 0], X_rec[0, 1])
    
    # 可视化
    plt.scatter(X[:, 0], X[:, 1], c='white', edgecolors='blue')
    plt.scatter(X_rec[:, 0], X_rec[:, 1], c='white', edgecolors='red')
    for i in range(m):
        plt.plot([X[i, 0], X_rec[i, 0]], [X[i, 1], X_rec[i, 1]], '--', c='black')
    plt.xticks(np.arange(-4, 4))
    plt.yticks(np.arange(-4, 4))
    plt.show()
    '''
    return U, S

def face():
    # PCA on  face data
    X = loadData('ex7faces.mat') # 5000*1024

    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(X[i, :].reshape(32, 32).T, cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()

    X_norm = featureNormalize(X)
    U, S = pca(X_norm)

    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.imshow(U[:, i].reshape(32, 32).T, cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()

    K = 100
    Z = np.dot(X, U[:, :K])  # m*k
    X_rec = np.dot(Z, U[:, :K].T)

    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(X_rec[i, :].reshape(32, 32).T, cmap='gray')
        plt.xticks(())
        plt.yticks(())
    plt.show()

if __name__ == '__main__':
    X = loadData('ex7data1.mat')
    X_norm = featureNormalize(X)
    #pca(X_norm)
    face()