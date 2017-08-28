# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:21:54 2017

@author: Qin
"""

import numpy as np
from scipy.optimize import fmin_cg
    
def initWeights(size):
    'init weights'
    # np.sqrt(6) / np.sqrt(size[0]+size[1])
    epsilon = 0.09
    weights = np.array([])
    for i in range(len(size)-1):
        weight = np.random.rand(size[i+1], size[i]+1) * 2 * epsilon - epsilon
        weights = np.hstack((weights, weight.flatten()))
    
    return weights
    
def feedForward(weights, a):
    'forward propagation'
    weigth1 = weights[:19625].reshape(25, 785)
    weight2 = weights[19625:].reshape(10, 26)
    for w in [weigth1, weight2]:
        z = np.dot(w, np.vstack((1, np.c_[a])))
        a = sigmoid(z)
    
    return a
    
def backward(weights, lam, X, y):
    'back propagation(gradient)'
    m = len(X)
    weight1 = weights[:19625].reshape(25, 785)
    weight2 = weights[19625:].reshape(10, 26)
    weight1_grad = np.zeros(weight1.shape)
    weight2_grad = np.zeros(weight2.shape)
    
    for i in range(m):
        a1 = np.vstack((1, np.c_[X[i]]))                
        z2 = np.dot(weight1, a1)                
        a2 = np.vstack((1, sigmoid(z2)))
        z3 = np.dot(weight2, a2)
        a3 = sigmoid(z3)
        
        delta_3 = a3 - np.c_[y[i]] # 10*1
        delta_2 = np.dot(weight2.T, delta_3)[1:] * sigmoid_prime(z2) # 25*1
        
        weight2_grad += np.dot(delta_3, a2.T)
        weight1_grad += np.dot(delta_2, a1.T)
        
    w1 = weight1.copy()
    w2 = weight2.copy()
    w1[:, 0] = 0
    w2[:, 0] = 0
    
    weight2_grad = 1 / m * weight2_grad + lam / m * w2
    weight1_grad = 1 / m * weight1_grad + lam / m * w1
    
    weight_grad = np.array([])
    weight_grad = np.hstack((weight_grad, weight1_grad.flatten()))
    weight_grad = np.hstack((weight_grad, weight2_grad.flatten()))
    
    return weight_grad
     
def predict(weights, testing_data):
    'predict'
    m = len(testing_data)
    result = []
    
    for i in range(m):
        a = feedForward(weights, np.c_[testing_data[i]])
        result.append([np.argmax(a)])
    
    return np.array(result)

def sigmoid(z):
    'sigmoid function'
    return 1.0 / (1 + np.exp(-1.0*z))
    
def sigmoid_prime(z):
    'derivative of sigmoid function'
    return sigmoid(z) * (1 - sigmoid(z))

def GK(weights, lam, X, y):
    'gradient checking'
    epsilon = 0.0001
    weights[0][10, 4] = weights[0][10, 4] - epsilon
    temp1 = costFunction(weights, lam, X, y)
    weights[0][10, 4] = weights[0][10, 4] + 2 * epsilon
    temp2 = costFunction(weights, lam, X, y)
    
    return np.abs((temp1-temp2)/(2*epsilon))

def costFunction(weights, lam, X, y):
    'cost function'
    m = len(X) 
    weight1 = weights[:19625].reshape(25, 785)
    weight2 = weights[19625:].reshape(10, 26)     
    J = 0
          
    for i in range(m):
        a = X[i]
        h = feedForward(weights, a)
        J += np.sum(np.c_[y[i]] * np.log(h) + (1 - np.c_[y[i]]) * np.log(1 - h))
    J = -1 / m * J + lam / (2 * m) * sum([np.sum(np.power(w[:, 1:], 2)) for w in [weight1, weight2]])
        
    return J

if __name__ == '__main__':
    X = np.load('train_data.npy')
    y = np.load('train_label.npy')    
    X_test = np.load('test_data.npy')
    y_test = np.load('test_label.npy')
    
    size = [784, 25, 10]
    lam = 0
    init_weights = initWeights(size)
    res = fmin_cg(costFunction, init_weights, fprime=backward, args=(lam, X, y))
   
    result = predict(res, X_test)
    p = (result == y_test).sum() / 200
    print('accurate: ', p)

   
