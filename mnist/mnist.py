#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/6/11 15:36
"""

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  显示所有信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  显示错误和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  显示错误信息
'''


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

'''
Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，
然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。）
'''

import tensorflow as tf

'''
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）
'''
# 通过操作符号变量来描述这些可交互的操作单元
x = tf.placeholder('float', [None, 784])  # 28x28 = 784

# 我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
# 模型也需要权重值和偏置量，
w = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])

'''
注意，W的维度是[784，10]，因为我们想要用784维的图片向量
    乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
b的形状是[10]，所以我们可以直接把它加到输出上面。
'''
# 可以定义我们的模型
y = tf.nn.softmax(tf.matmul(x, w) + b)

'''
    训练模型：
        为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。
        其实，在机器学习，我们通常定义指标来表示一个模型是坏的，
        这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。
        但是，这两种方式是相同的。
    一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）
'''

# 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
y_ = tf.placeholder('float', [None, 10])
# 计算交叉熵:
cross_entropy = tf.reduce_sum(y_ * tf.log(y))

'''
因为TensorFlow拥有一张描述你各个计算单元的图，
它可以自动地使用反向传播算法(backpropagation algorithm)
来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
'''
# 在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）
# 以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # 然后开始训练模型，这里我们让模型循环训练1000次
    for _ in range(1000):
        batch_xs, batch_ys = train_step
