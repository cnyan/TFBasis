#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/4/2 15:09
Describe：
    实现一个简易的网络模型，实现了通过集合计算一个 4 层全连接神经网络带L2正则化损失函数的功能
    
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义训练轮数
training_steps = 30000
# 定义带标签的输入数据，并在for循环内进行填充
data = []
label = []
for i in range(200):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)
    # 对x1 和 x2进行判断，如果产生的点落在半径为1的圆内，则label的值为0，否则为1
    if x1 ** 2 + x2 ** 2 <= 1:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

# 以-1在这里应该可以理解为一个正整数通配符，它代替任何整数,最终将data 变成一个n 行2列的二维数组
data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)


# 定义完成前向传播的隐含层
def hidden_layer(input_tensor, weight1, bias1, weight2, bias2, weight3, bias3):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
    layer3 = tf.nn.relu(tf.matmul(layer2, weight3) + bias3)
    return layer3


x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')  # 衡量值

# 定义权重参数和偏置参数
weight1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
weight2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
weight3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1]))

# data数组长度
sample_size = len(data)
# 得到隐藏层前向传播结果
y = hidden_layer(x, weight1, bias1, weight2, bias2, weight3, bias3)  # 实际值
# 自定义损失函数
error_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
tf.add_to_collection('losses', error_loss)  # 加入集合操作

# 在权重上实现L2正则化,正则化作用于损失函数
regularizer = tf.contrib.layers.l2_regularizer(0.01)
regularization = regularizer(weight1) + regularizer(weight2) + regularizer(weight3)
tf.add_to_collection("losses", regularization)  # 加入集合操作

# 获取所有损失值，并在add_n()函数中进行加和运算
loss = tf.add_n(tf.get_collection('losses'))

# 定义一个优化器，优化损失函数的最终取值
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
#    print(weight1)
#    print(weight1.eval())
    # 在for循环中进行30000轮训练
    for i in range(training_steps):
        sess.run(train_op, feed_dict={x: data, y_: label})

        # 每隔2000次，就输出loss的值
        if i % 2000 == 0:
            loss_value = sess.run(loss, feed_dict={x: data, y_: label})
            print("after %d steps,mse_loss:%f" % (i, loss_value))


