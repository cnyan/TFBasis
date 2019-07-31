# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/26 15:03
@Describe：
    conv2d() 卷积操作
"""

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 输入I
M = np.array([[[2], [1], [2], [-1]], [[0], [-1], [3], [0]], [[-2], [1], [-1], [4]], [[-2], [0], [-3], [4]]],
             dtype='float32').reshape(1, 4, 4, 1)
# 过滤器K
filter_weight = tf.compat.v1.get_variable('weights', [2, 2, 1, 1],
                                          initializer=tf.constant_initializer([[-1, 4], [2, 1]]))
# 过滤器偏置项，过滤器深度为1 等于 神经网络下一层的深度
biases = tf.compat.v1.get_variable("biases", [1], initializer=tf.constant_initializer(1))

# 定义输入
x = tf.compat.v1.placeholder('float32', [1, None, None, 1])
# 使用conv2d()函数实现卷积层前向传播的算法 SAME 表示全0填充，VALID表示不使用0填充
conv = tf.nn.conv2d(x, filter=filter_weight, strides=[1, 1, 1, 1], padding='SAME')
# 添加偏置项
add_bias = tf.nn.bias_add(conv, biases)

init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    init_op.run()
    M_conv = sess.run(add_bias, feed_dict={x: M})

    print("M after convolution: \n", M_conv)
