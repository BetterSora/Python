import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    # 加载数据
    mat = sio.loadmat(filename)
    X = mat['X']
    # cv set
    Xval = mat['Xval']
    # cv truth
    yval = mat['yval']

    return X, Xval, yval

def estimateGaussian(X):
    # 求出每个特征的mu和sigma2
    X_mean = np.mean(X, axis=0)
    X_sigma2 = np.var(X, axis=0)

    return X_mean, X_sigma2

def multivariateGaussian(X, mu, sigma2):
    # 多元高斯计算概率P
    # 这所以这样写协方差矩阵是因为默认不同维度之间相互独立
    n = len(mu)
    cov_mat = np.diag(sigma2)
    p = np.diag(1 / (np.power(2*np.pi, n/2) * np.power(np.linalg.det(cov_mat), 1/2)) * np.exp(-1/2*(X-mu).dot(np.linalg.inv(cov_mat)).dot((X-mu).T)))

    return p

def visualizeFit(X, mu, sigma2):
    # 可视化
    x1 = np.linspace(0, 35, 71)
    x2 = np.linspace(0, 35, 71)
    xx, yy = np.meshgrid(x1, x2)
    p = multivariateGaussian(np.c_[xx.ravel(), yy.ravel()], mu, sigma2)

    plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolors='black')
    c = plt.contour(xx, yy, p.reshape(xx.shape), [10**x for x in range(-20, 0, 3)])
    plt.clabel(c, inline=True, fontsize=10)
    plt.xticks(np.arange(0, 36, 5))
    plt.yticks(np.arange(0, 36, 5))
    plt.show()

def selectThreshold(yval, pval):
    # 选择阈值
    bestEpsilon = 0
    bestF1 = 0
    stepsize = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), stepsize):
        predict = np.where(pval<epsilon, 1, 0)
        TP = np.sum(yval[predict==1]==1)
        TN = np.sum(yval[predict==0]==0)
        FP = np.sum(yval[predict==1]==0)
        FN = np.sum(yval[predict==0]==1)

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2*P*R / (P+R)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestF1, bestEpsilon

if __name__ == '__main__':
    X, Xval, yval = loadData('ex8data1.mat')
    mu, sigma2 = estimateGaussian(X)
    #p = multivariateGaussian(X, mu, sigma2)
    #visualizeFit(X, mu, sigma2)

    pval = multivariateGaussian(Xval, mu, sigma2)
    bestF1, bestEpsilon = selectThreshold(yval, pval)
    print('Best epsilon found using cross-validation: %e' % bestEpsilon)
    print('Best F1 on Cross Validation Set:  %f' % bestF1)

    X, Xval, yval = loadData('ex8data2.mat')
    mu, sigma2 = estimateGaussian(X)
    pval = multivariateGaussian(Xval, mu, sigma2)
    bestF1, bestEpsilon = selectThreshold(yval, pval)
    print('Best epsilon found using cross-validation: %e' % bestEpsilon)
    print('Best F1 on Cross Validation Set:  %f' % bestF1)
