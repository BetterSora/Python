import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def loadData():
    # 加载数据
    mat = sio.loadmat('ex7data2.mat')

    return  mat['X']

def initialCentroids(data, K):
    # 初始化簇心
    centroids = []
    for i in range(K):
        index = np.random.randint(0, len(data))
        centroids.append(data[index])

    return centroids

def getDistance(a, b):
    # 欧氏距离
    dst = np.power(a-b, 2).sum()

    return dst

def findClosestCentroids(old_centroids, data):
    # 簇分配
    label = []

    for x in data:
        dst = [getDistance(np.array(centroid), np.array(x)) for centroid in old_centroids]
        label.append(np.argmin(dst))

    return np.array(label)

def computeCentroids(X, label, K):
    # 计算簇心
    new_centroids = []
    for i in range(K):
        idx = np.where(label==i)[0]
        new_centroids.append(np.mean(X[idx, :], axis=0))

    return new_centroids

def costFunction(centroids, label, X):
    # 代价函数
    cost = 0
    for i in range(len(centroids)):
        cost += np.power(X[np.where(label==i), :]-centroids[i], 2).sum() / len(X)

    return cost

def kmeans(X, max_iters, K):
    # kmeans
    centroids = initialCentroids(X, K)

    for i in range(max_iters):
        print('iter:', i+1)
        label = findClosestCentroids(centroids, X)
        centroids = computeCentroids(X, label, K)

    return label, centroids

def imageCompression():
    # kmeans图片压缩
    #imageArray = plt.imread('bird_small.png')
    imageArray = plt.imread('1.jpg')
    plt.subplot(121)
    plt.imshow(imageArray)

    imageShape = imageArray.shape
    X = imageArray.reshape(imageShape[0]*imageShape[1], imageShape[2])

    label, centroids = kmeans(X, 10, 16)

    for i in range(len(centroids)):
        X[np.where(label==i)[0], :] = centroids[i]

    plt.subplot(122)
    plt.imshow(X.reshape(imageShape[0], imageShape[1], imageShape[2]))
    plt.show()

if __name__ == '__main__':
    '''
    X = loadData()

    # kmeans
    label, centroids = kmeans(X, 10, 3)
    cost = costFunction(centroids, label, X)
    print('cost: ', cost)

    # 可视化
    colors = ['red', 'green', 'yellow', 'pink']
    for i in range(len(centroids)):
        data = X[np.where(label==i)[0], :]
        plt.scatter(data[:, 0], data[:, 1], c=colors[i], edgecolors='black')
        plt.scatter(centroids[i][0], centroids[i][1], marker='+', c='blue')
    plt.show()
    '''
    imageCompression()
