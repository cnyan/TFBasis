#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/4/2 13:22
Describe：
    优化器:使用优化算法去优化损失函数可以逐渐减少训练误差
        优化器的学习率需要独立设置
        调用minimize()函数来指定最小化的损失函数
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_step = tf.Variable(0)
# 指数衰减的学习率：设定初始学习率0.8，因为指定了staircase=True，所以每训练100次，学习率乘以0.9
decay_learning_rate = tf.train.exponential_decay(learning_rate=0.8, global_step=training_step, decay_steps=100,
                                                 decay_rate=0.9, staircase=True)
# 损失函数loss 是目标函数
# learning_step = tf.train.GradientDescentOptimizer(learning_rate=decay_learning_rate).minimize(loss=loss, global_step=training_step)

# ========================================== 正则化
weight = tf.constant([[1.0, 2.0], [-3.0, -4.0]])

# .5 表示正则化项的权重，对应于公式 J(w) + λR(w) 中的λ,通常去一个非常小的数字，例如0.01
regularizer_l2 = tf.contrib.layers.l2_regularizer(.5)  # 返回一个函数
regularizer_l1 = tf.contrib.layers.l1_regularizer(.5)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(regularizer_l1(weight)))
    print(sess.run(regularizer_l2(weight)))
