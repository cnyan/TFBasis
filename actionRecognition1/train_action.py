# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/20 11:04
@Describe：
    基于多层感知机 两层深度网络实现训练动作分类
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from actionRecognition1 import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 数据处理
actions_data = input_data.read_data_sets('src/actions_data.npy', test_size=0.1)

'''定义相关参数(超参数)'''
batch_size = 100  # 设置每一轮训练的batch大小
learning_rate = 0.8  # 设置学习率
learning_rate_decay = 0.999  # 学习率的衰减系数,一般设置为非常接近于1的值
max_steps = 30000  # 最大训练步数
training_step = tf.Variable(0, trainable=False)

'''定义全连接的前向网络，定义得到一个隐藏层和输出层的前向传播计算方式，激活函数用relu'''


def hidden_layer(input_tensor, weights1, bias1, weights2, bias2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
    outLayer = tf.matmul(layer1, weights2) + bias2
    return outLayer


import numpy as np


def next_batch_data(batch_step, batch_size, actions_set):
    actions_data = actions_set['train_data']
    actions_label = actions_set['train_label']

    complex_array = np.concatenate([actions_data, actions_label], axis=1)
    if ((batch_step + 1) * batch_size) < len(actions_data):

        return complex_array[batch_step * batch_size:(batch_step + 1) * batch_size, :-10], \
               complex_array[batch_step * batch_size:(batch_step + 1) * batch_size, -10:]
    else:
        np.random.shuffle(complex_array)
        return complex_array[: batch_size, :-10], complex_array[: batch_size, -10:]


# 开始
x = tf.compat.v1.placeholder(tf.float32, [None, 1890], name='x-input')  # 输入数据
y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y-output')  # 标签

# 生成隐藏层参数，隐藏层有1890个神经单元，其中weights包含1890*500个参数
weights1 = tf.Variable(tf.random.truncated_normal([1890, 500], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[500]))
# 生成输出层参数，输出层10个神经元，weights2包含500*10 = 5000个参数
weights2 = tf.Variable(tf.random.truncated_normal([500, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))

# 开始计算经过神经网络前向传播后得到的y值
y = hidden_layer(x, weights1, bias1, weights2, bias2, 'y')

'''为了提高采用随机梯度下降算法的训练神经网络得到的模型在测试集上的表现，
    使用“滑动平均模型”应用到隐藏层与输出层的权重参数与偏置参数
    使用“滑动平均模型”计算影子变量
'''
# 初始化一个"滑动平均类"，衰减率为0.99
# 为了使这个训练前期可以更新得更快，这里提供了num_updates参数，并设置为当前网络的训练轮数
averages_class = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=training_step)
# 定义一个更新变量滑动平均值的操作，需要向滑动平均类的apply()函数提供一个参数列表
# train_variables() 函数返回集合图上 Graph.TRAINABLE_VARIABLES中的元素，这个集合的元素就是所有没有指定trainable=false的元素
averages_op = averages_class.apply(tf.compat.v1.trainable_variables())  # 应用到集合图上所有权重参数和偏置参数，随着每次迭代动态改变参数值

# 再次计算经过神经网络前向传播后得到的y的值，这里使用了滑动平均，但要牢记滑动平均值只是一个影子变量
average_y = hidden_layer(x, weights1, bias1, weights2, bias2, 'average_y')

'''开始计算交叉熵损失函数'''
# 计算交叉熵函数J（w），logits是前向传播的结果，axis=1 按行获取最大值的下标，即对应识别的结果
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, axis=1))
# 正则化 λR(w)
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
regularization = regularizer(weights1) + regularizer(weights2)
# 总的损失函数= J（w）+ λR(w)
loss = tf.reduce_mean(cross_entropy) + regularization

''' 使用随机梯度下降SGD优化器，学习率采用指数衰减的方式
    learning_rate: 学习率 0.8
    global_step：当前训练次数
    decay_steps：衰减速度
    decay_rate：衰减系数，小于1的较大数 0.99
'''
# 设置指数衰减法来设置学习率，staircase=false,即学习率连续衰减
learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=learning_rate, global_step=training_step,
                                                     decay_steps=100, decay_rate=learning_rate_decay, staircase=True)
train_op_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                                  global_step=training_step)

'''
    在训练这个模型时，每过一遍数据，既要通过反向传播来更新神经网络中的参数，又需要更新每一个参数的滑动平均值
    control_dependencies()用于完成这样的一次性的多次操作
    # 下面两行代码，可以用一行代码替换
    # train_op = tf.group(train_op_step,averages_op)
'''
with tf.control_dependencies([train_op_step, averages_op]):
    train_op = tf.no_op(name='train')  # 什么都不做，仅做为点位符使用控制边界。

'''检查使用了滑动平均值模型的神经网络前向传播是否正确，平均值就是模型在这一组数据上的正确率，求准确率'''
current_prediction = tf.equal(tf.argmax(average_y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))  # tf.cast()这里将布尔类型变为float32类型的数据

''' 构建会话 '''
init_op = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init_op)

    # 准备验证数据
    validate_feed = {x: actions_data['validation_data'], y_: actions_data['validation_label']}
    # 准备测试数据
    test_feed = {x: actions_data['test_data'], y_: actions_data['test_label']}

    for i in range(max_steps):
        # 每训练1000次计算滑动模型在验证数据集上的结果
        if i % 1000 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print('After % d training step(s),validation accuracy using average model is %g%%' % (
                i, validate_accuracy * 100))

        # 产生这一轮的batch训练数据，并进行训练
        xs, ys = next_batch_data(batch_step=i, batch_size=batch_size, actions_set=actions_data)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    # 得到正确率
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s)，test accuracy using average model is %g%%" % (max_steps, test_accuracy * 100))
