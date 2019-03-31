#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/3/31 15:22
Describe：
    深度前馈神经网络
    
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 输入
x = tf.constant([0.9, 0.85], shape=[1, 2])
# 权重
w1 = tf.Variable(tf.constant([[0.2, 0.1, 0.3], [0.2, 0.4, 0.3]], shape=[2, 3], name='w1'))
w2 = tf.Variable(tf.constant([0.2, 0.5, 0.25], shape=[3, 1], name='w2'))
# 偏置项
b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name='b1')
b2 = tf.constant([-0.3], shape=[1], name='b2')
# 初始化全部变量
init_op = tf.global_variables_initializer()
# 乘法操作(线性模型)---前向传播过程
a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2
# 加入激活函数
aa = tf.nn.relu(tf.matmul(x, w1) + b1)
yy = tf.nn.relu(tf.matmul(aa, w2) + b2)
# 定义交叉熵函数
cross_entropy = -tf.reduce_mean(yy * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# where() 及 greater()
q1 = tf.constant([1.0, 2.0, 3.0, 4.0])
q2 = tf.constant([6.0, 7.0, 4.0, 3.0])

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
    print(sess.run(yy))
    print(sess.run(cross_entropy))
    print('====== where() 及 greater()=======')
    print(sess.run(tf.greater(q1, q2)))
    print(tf.where(tf.greater(q1, q2), q1, q2).eval())
    print('=========end==============')
