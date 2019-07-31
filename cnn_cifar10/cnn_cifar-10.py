# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/26 16:48
@Describe：
    构造循环神经网络的整体结构,并进行训练和测试过程
    卷积、池化、卷积、池化、拉直数据、三个全连接、损失函数
"""
import tensorflow as tf
import numpy as np
import time
import math
import os
from cnn_cifar10 import cifar10_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

max_steps = 4000
batch_size = 128
num_examples_for_eval = 10000
data_dir = '/src/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape, stddev))
    if w1 is not None:
        # 给权重var加一个正则化
        # multiply()函数原型multiply(x,y,name)
        weight_loss = tf.multiply(tf.compat.v1.nn.l2_loss(var), w1, name='weight_loss')
        tf.compat.v1.add_to_collection("losses", weight_loss)
    return var


# 对于用于训练的图片数据，distorted参数为True，表示进行数据增强处理
images_train, labels_train = cifar10_data.inputs(data_dir, batch_size=batch_size, distorted=True)
# 对于用于测试的图片数据，distorted参数为Nnone，表示不进行数据增强处理
images_test, labels_test = cifar10_data.inputs(data_dir, batch_size=batch_size, distorted=None)

# 创建placeholder
x = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, 24, 24, 3])
y_ = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size])

# 第一个卷积层:5*5的卷积核，输入通道3，输出深度64，使用variable_with_weight_loss创建卷积核
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# 使用conv2d()函数实现卷积层前向传播的算法
conv1 = tf.compat.v1.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[64]))
relu1 = tf.compat.v1.nn.relu(tf.nn.bias_add(conv1, bias=bias1))
# 池化操作，池化核3*3，步长为2*2
pool1 = tf.compat.v1.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二个卷积层：
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.compat.v1.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[64]))
relu2 = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(conv2, bias2))
pool2 = tf.compat.v1.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 拉直数据
reshape = tf.compat.v1.reshape(pool2, [batch_size, -1])
# 获取长度
dim = reshape.get_shape()[1].value

# 第一个全连接层,隐藏单元数 384，
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[384]))
fc_1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(reshape, weight1) + fc_bias1)

# 第二个全连接层,隐藏单元192个
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[192]))
fc_2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(fc_1, weight2) + fc_bias2)

# 第三个全连接层,隐藏单元10个
weight2 = variable_with_weight_loss(shape=[192, 10], stddev=(1 / 192.0), w1=0.0)
fc_bias3 = tf.compat.v1.Variable(tf.compat.v1.constant(0.0, shape=[10]))
result = tf.compat.v1.add(tf.compat.v1.matmul(fc_2, weight2), fc_bias3)

# 计算损失函数:包括权重参数的正则化损失和交叉熵损失
cross_entopy = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
                                                                        labels=tf.cast(y_, tf.compat.v1.int64))
weights_with_l2_loss = tf.compat.v1.add_n(tf.compat.v1.get_collection('losses'))
loss = tf.compat.v1.reduce_mean(cross_entopy) + weights_with_l2_loss

# 优化器，全局学习率0.001
train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)

# 输出结果中 top k 的准确率,k=1 即top(1) 即准确率最高的时候
top_k_op = tf.compat.v1.nn.in_top_k(result, y_, k=1)
init_op = tf.compat.v1.global_variables_initializer()

# 创建会话
with tf.compat.v1.Session() as sess:
    init_op.run()

    # 开启多线程
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(max_steps):
        start_time = time.time()
        # 获取一个batch 的数据
        image_batch, label_batch = sess.run([images_train, labels_train])
        # 损失函数和优化器
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})

        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            # 打印每一轮训练的耗时
            print("step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)" %
                  (step, loss_value, examples_per_sec, sec_per_batch))

    # 测试集：
    # math.ceil()函数用于求整，原型为ceil(x)
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环内统计所有预测正确的样例的个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: images_test, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
