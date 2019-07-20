#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/6/11 16:10
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    import tensorflow.examples.tutorials.mnist.input_data as input_data

    mnist = input_data.read_data_sets("download", one_hot=True)
    # 查看数据维度
    print('Training data and label size:')
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.train.images.shape, mnist.train.labels.shape)

    print('Testing data and label size:')
    print(mnist.test.images.shape, mnist.test.labels.shape)

    print('validating data and label size:')
    print(mnist.validation.images.shape, mnist.validation.labels.shape)

    print('Example training data:{}'.format(mnist.train.images[0]))
    print('Example training label:{}'.format(mnist.train.labels[0]))


