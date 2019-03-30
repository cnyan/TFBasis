#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/6/12 11:15

"""
'''
1. Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。
'''
import numpy as np


# (正则化后的代价函数)代价函数
def costFunction(initial_theta, X, y, inital_lambda):
    m = len(y)
    J = 0

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()  # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1), theta1)  # transpose 矩阵转置
    # 正则化的代价方程
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y),
                                                      np.log(1 - h)) + temp * inital_lambda / 2) / m
    return J


# end

# 计算梯度(正则化后的代价的梯度)
def gradient(initial_theta, X, y, inital_lambda):
    m = len(y)
    grad = np.zeros(initial_theta.shape[0])

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    # 正则化的梯度
    grad = np.dot(np.transpose(X), h - y) / m + inital_lambda / m * theta1
    return grad


# S 型函数（Sigmoid函数）
def sigmoid(z):
    h = np.zeros((len(z), 1))

    h = 1.0 / (1.0 + np.exp(-z))
    return h


# 映射为多项式(映射为2次方的形式为)
def mapFeature(x1, x2):
    degree = 3  # 映射的最高次方
    out = np.ones((x1.shape[0], 1))  # 映射后的结果数组（取代x)

    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            # 矩阵直接乘相当于matlab中的点乘.*
            temp = x1 ** (1 - j) * (x2 ** j)
            # hstack(tup) 它其实就是水平(按列顺序)把数组给堆叠起来
            out = np.hstack((out, temp.reshape(-1, -1)))
    return out


'''
使用scipy的优化方法
梯度下降使用scipy中optimize中的fmin_bfgs函数
initial_theta表示初始化的值，
fprime指定costFunction的梯度
args是其余参数，以元组的形式传入，最后会将最小化costFunction的theta返回

result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,y,initial_lambda)) 
'''

# 使用scikit-learn库中的逻辑回归模型实现
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def logisticRegression():
    data = loadtxtAndcsv_data('src/data1.txt', ',', np.float64)
    X = data[:, 0, -1]
    y = data[:-1]

    # 划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def loadtxtAndcsv_data(fileName, split, dataType):
    pass
