import scipy.io as sio
import numpy as np
from scipy.optimize import fmin_cg

def loadData(filename):
    # 加载数据
    mat = sio.loadmat(filename)
    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
    Y = mat['Y']
    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
    R = mat['R']

    return Y, R

def cofiCostFunc(parameters, num_users, num_features, num_movies, Y, R, lam):
    # 协同过滤算法代价函数
    # theta: num_users * num_features X: num_movies * num_features
    X = parameters[0:num_movies*num_features].reshape(num_movies, num_features)
    theta = parameters[num_movies*num_features:].reshape(num_users, num_features)
    J = 1/2 * np.sum(R * np.power(np.dot(X, theta.T)-Y, 2)) + lam/2 * np.sum(np.power(X, 2)) + lam/2 * np.sum(np.power(theta, 2))

    return J

def gradient(parameters, num_users, num_features, num_movies, Y, R, lam):
    # 代价函数的梯度
    X = parameters[0:num_movies * num_features].reshape(num_movies, num_features)
    theta = parameters[num_movies*num_features:].reshape(num_users, num_features)

    grad_X = np.zeros(X.shape); grad_theta = np.zeros(theta.shape)

    for i in range(num_movies):
        grad_X[i, :] = np.sum(np.c_[R[i, :]*(np.dot(theta, X[i, :].T) - Y[i, :])]*theta, axis=0) + lam * X[i, :]

    for j in range(num_users):
        grad_theta[j, :] = np.sum(np.c_[R[:, j]*(np.dot(theta[j, :], X.T) - Y[:, j])]*X, axis=0) + lam * theta[j, :]
    '''
    for i in range(num_movies):
        for j in range(num_users):
            if R[i, j] == 1:
                grad_X[i, :] += (np.dot(theta[j, :], X[i, :].T) - Y[i, j]) * theta[j, :]
        grad_X[i, :] += lam * X[i, :]

    for j in range(num_users):
        for i in range(num_movies):
            if R[i, j] == 1:
                grad_theta[j, :] += (np.dot(theta[j, :], X[i, :].T) - Y[i, j]) * X[i, :]
        grad_theta[j, :] += lam * theta[j, :]
    '''
    grad = np.hstack((grad_X.flatten(), grad_theta.flatten()))

    return grad

def loadMovieList():
    # 加载电影名称列表
    movieList = []

    with open('movie_ids.txt', encoding='iso-8859-1') as f:
        for line in f:
            movieList.append(line[2:-1])

    return movieList

def normalizeRatings(Y):
    # 均值归一化
    Y_mean = np.mean(Y, axis=1)
    Y_norm = Y - np.c_[Y_mean]

    return Y_norm, Y_mean

def test():
    # 测试代价函数是否正确
    mat = sio.loadmat('ex8_movieParams.mat')
    X = mat['X']
    theta = mat['Theta']
    Y, R = loadData('ex8_movies.mat')

    # lambda=0
    num_users = 4
    num_movies = 5
    num_features = 3
    J = cofiCostFunc(np.hstack((X[0:num_movies, 0:num_features].flatten(), theta[0:num_users, 0:num_features].flatten())),
                     num_users, num_features, num_movies, Y[0:num_movies, 0:num_users], R[0:num_movies, 0:num_users], 0)
    print('Cost at loaded parameters: %f (this value should be about 22.22)' % J)

    # lambda=1.5
    J = cofiCostFunc(np.hstack((X[0:num_movies, 0:num_features].flatten(), theta[0:num_users, 0:num_features].flatten())),
                     num_users, num_features, num_movies, Y[0:num_movies, 0:num_users], R[0:num_movies, 0:num_users], 1.5)
    print('Cost at loaded parameters: %f (this value should be about 31.34)' % J)

    # 推荐电影
    movieList = loadMovieList()
    # 初始化用户评分
    my_ratings = np.zeros((len(movieList), 1), dtype=np.int64)
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    # 将新用户数据添加到原始数据中
    Y = np.c_[Y, my_ratings]
    R = np.c_[R, np.where(my_ratings!=0, 1, 0)]

    # 为了给没有任何电影评价的用户推荐电影，进行均值归一化，否则该用户的电影评分都将为0
    Y_norm, Y_mean = normalizeRatings(Y)

    # 初始化参数
    num_movies = Y_norm.shape[0]
    num_users = Y_norm.shape[1]
    num_features = 10

    X = np.random.randn(num_movies, num_features)
    theta = np.random.randn(num_users, num_features)

    initial_parameters = np.hstack((X.flatten(), theta.flatten()))
    result = fmin_cg(cofiCostFunc, initial_parameters, fprime=gradient, args=(num_users, num_features, num_movies, Y_norm, R, 10), maxiter=100)

    X = result[0:num_movies * num_features].reshape(num_movies, num_features)
    theta = result[num_movies * num_features:].reshape(num_users, num_features)
    p = np.dot(X, theta.T)
    my_predictions = p[:, -1] + Y_mean
    print(np.sort(my_predictions))