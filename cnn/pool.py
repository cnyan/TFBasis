# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/26 16:18
@Describe：
    池化层的实现
"""

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

M = np.array([[[-2], [2], [0], [3]], [[1], [2], [-1], [2]], [[0], [-1], [1], [0]]], dtype='float32').reshape(1, 3, 4, 1)

filter_weight = tf.compat.v1.get_variable("weights", [2, 2, 1, 1],
                                          initializer=tf.compat.v1.constant_initializer([[2, 0], [-1, 1]]))
biases = tf.compat.v1.get_variable("biases", [1], initializer=tf.compat.v1.constant_initializer(1))

x = tf.compat.v1.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter=filter_weight, strides=[1, 1, 1, 1], padding='SAME')

add_bias = tf.nn.bias_add(conv, biases)

# 池化操作:聚合统计操作，降低特征参数 ksize为池化核的大小，常用为2*2，或3*3
pool = tf.nn.max_pool2d(add_bias, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    init_op.run()
    M_conv = sess.run(add_bias, feed_dict={x: M})
    M_pool = sess.run(pool, feed_dict={x: M})
    print('after max pooled:\n', M_pool)
