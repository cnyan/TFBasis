#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2019/4/11 15:02
Describe：
    训练mnist数据集
    
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''获取数据'''
mnist = input_data.read_data_sets("download", one_hot=True)

'''定义相关参数'''
batch_size = 100  # 设置每一轮训练的batch大小
learning_rate = 0.8  # 设置学习率
learning_rate_decay = 0.999  # 学习率的衰减,一般设置为非常接近于1的值
max_steps = 30000  # 最大训练步数
training_step = tf.Variable(0, trainable=False)

'''定义全连接的前向网络，定义得到一个隐藏层和输出层的前向传播计算方式，激活函数用relu'''


def hidden_layer(input_tensor, weights1, bias1, weights2, bias2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
    outLayer = tf.matmul(layer1, weights2) + bias2
    return outLayer


x = tf.placeholder(tf.float32, [None, 784], name='x-input')  # 输入数据
y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')  # 标签

# 生成隐藏层参数，隐藏层有784个神经单元，其中weights包含784*500=392000个参数
weights1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[500]))
# 生成输出层参数，输出层10个神经元，weights2包含500*10 = 5000个参数
weights2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
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
averages_op = averages_class.apply(tf.trainable_variables())  # 应用到集合图上所有权重参数和偏置参数，随着每次迭代动态改变参数值

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

''' 使用随机梯度下降SGD优化器，学习率采用指数衰减的方式'''
# 设置指数衰减法来设置学习率，staircase=false,即学习率连续衰减
learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=training_step,
                                           decay_steps=mnist.train.num_examples / batch_size,
                                           decay_rate=learning_rate_decay, staircase=False)
train_op_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
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
crorent_prediction = tf.equal(tf.argmax(average_y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(crorent_prediction, tf.float32))  # tf.cast()这里将布尔类型变为float32类型的数据

''' 构建会话 '''
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # 准备验证数据
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):
        # 每训练1000次计算滑动模型在验证数据集上的结果
        if i % 1000 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print('After % d training step(s),validation accuracy using average model is %g%%' % (
                i, validate_accuracy * 100))
        # 产生这一轮的batch训练数据，并进行训练
        xs, ys = mnist.train.next_batch(batch_size=100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    # 得到正确率
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s)，test accuracy using average model is %g%%" % (max_steps, test_accuracy * 100))
