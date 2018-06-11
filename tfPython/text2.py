#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/6/11 15:36
"""

'''
    手写数字识别 (MNIST)
    train-images-idx3-ubyte.gz:训练集图片 - 55000 张 训练图片, 5000 张 验证图片
    train-labels-idx1-ubyte.gz:训练集图片对应的数字标签
    t10k-images-idx3-ubyte.gz:测试集图片 - 10000 张 图片
    t10k-labels-idx1-ubyte.gz:测试集图片对应的数字标签
'''
'''
    底层的源码将会执行下载、解压、重构图片和标签数据来组成以下的数据集对象:
    data_sets.train : 55000 组 图片和标签, 用于训练。
    data_sets.validation: 5000 组 图片和标签, 用于迭代验证训练的准确性。
    data_sets.test：10000 组 图片和标签, 用于最终测试训练的准确性。
'''

from tfPython import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
