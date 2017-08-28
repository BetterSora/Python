# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:49:14 2017

@author: Qin
"""

import numpy as np

class Network(object):
    def __init__(self, size):
        'init network'
        self.size = size
        self.num_layers = len(size)
        self.lam = 1
    
    def initWeights(self):
        'init weights'
        epsilon = np.sqrt(6) / np.sqrt(self.size[0]+self.size[1]) 
        self.weights = []
        for i in range(self.num_layers-1):
            weight = np.random.rand(self.size[i+1], self.size[i]+1) * 2 * epsilon - epsilon
            self.weights.append(weight)
    
    def feedForward(self, a):
        'forward propagation'
        for w in self.weights:
            z = np.dot(w, np.vstack((1, np.c_[a])))
            a = sigmoid(z)
        
        return a
    
    def backward(self, X, y):
        'back propagation(gradient)'
        m = len(X)
        weight1_grad = np.zeros(self.weights[0].shape)
        weight2_grad = np.zeros(self.weights[1].shape)
        
        for i in range(m):
            a1 = np.vstack((1, np.c_[X[i]]))                
            z2 = np.dot(self.weights[0], a1)                
            a2 = np.vstack((1, sigmoid(z2)))
            z3 = np.dot(self.weights[1], a2)
            a3 = sigmoid(z3)
            
            delta_3 = a3 - np.c_[y[i]] # 10*1
            delta_2 = np.dot(self.weights[-1].T, delta_3)[1:] * sigmoid_prime(z2) # 25*1
            
            weight2_grad += np.dot(delta_3, a2.T)
            weight1_grad += np.dot(delta_2, a1.T)
            
        weigth1 = self.weights[0].copy()
        weigth2 = self.weights[1].copy()
        weigth1[:, 0] = 0
        weigth2[:, 0] = 0
        
        weight2_grad = 1 / m * weight2_grad + self.lam / m * weigth2
        weight1_grad = 1 / m * weight1_grad + self.lam / m * weigth1
        
        return weight1_grad, weight2_grad
    
    def fit(self, X, y, epochs, eta):
        'gradient descent'
        print('start training')
        
        for i in range(epochs):
            weight1_grad, weight2_grad = self.backward(X, y)
            # gradient checking
            #temp = GK(self.weights, self.lam, X, y)
            #print(np.abs(np.abs(weight1_grad[10, 4])-temp))
            self.weights[0] -= eta * weight1_grad
            self.weights[1] -= eta * weight2_grad
            if i % 10 == 0:
                J = costFunction(self.weights, self.lam, X, y)
                print('epochs: %d cost: %f' %(i, J))    
         
    def predict(self, testing_data):
        'predict'
        m = len(testing_data)
        result = []
        
        for i in range(m):
            a = self.feedForward(np.c_[testing_data[i]])
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
    J = 0
          
    for i in range(m):
        h = X[i]
        for w in weights:
            z = np.dot(w, np.vstack((1, np.c_[h])))
            h = sigmoid(z)
        J += np.sum(np.c_[y[i]] * np.log(h) + (1 - np.c_[y[i]]) * np.log(1 - h))
    J = -1 / m * J + lam / (2 * m) * sum([np.sum(np.power(w[:, 1:], 2)) for w in weights])
        
    return J

if __name__ == '__main__':
    X = np.load('train_data.npy')
    y = np.load('train_label.npy')    
    X_test = np.load('test_data.npy')
    y_test = np.load('test_label.npy')
    
    network = Network([784, 25, 10])
    network.initWeights()
    network.fit(X, y, 200, 1.0)
    print('end')
    result = network.predict(X_test)
    p = (result == y_test).sum() / 200
    print('accurate: ', p)
    