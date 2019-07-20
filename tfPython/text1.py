#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/6/11 10:13
"""

'''                 
使用 TensorFlow, 你必须明白 TensorFlow:

    使用图 (graph) 来表示计算任务.图中的节点被称之为 op (operation 的缩写).  
    在被称之为 会话 (Session) 的上下文 (context) 中执行图.
    使用 tensor 表示数据.
    通过 变量 (Variable) 维护状态.
    使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

'''
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
  默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 为
'''

# '创建一个常量 V1 (图graph)，它是一个1行2列的矩阵')
v1 = tf.constant([[2, 3]])
print(v1)

# '创建一个常量 V2，它是一个2行1列的矩阵')
v2 = tf.constant([[2], [3]])
print(v2)

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(v1, v2)

print('打印 product 得到的不是乘法之后的结果，而是乘法本身')
print(product)

'''
# 定义一个会话
#  启动默认图.
sess = tf.Session()
# 运算乘法，得到一个结果 
result = sess.run(product)
print('打印会话结果')
print(result)
# 关闭会话
sess.close()
'''
with tf.Session() as sees:
    result = sees.run(product)
    print(result)

print('======Variable========' * 4)
'''
    变量： 通过 变量 (Variable) 维护状态.
        Variables for more details. 变量维护图执行过程中的状态信息. 
        下面的例子演示了如何使用变量实现一个简单的计数器.
'''
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name='counter')

# 创建一个op ，作用是使state加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 构造初始化，启动图后，变量必须先经过初始化（init) op 操作
# 必须增加一个初始化 op操作到默认图中
init_op = tf.global_variables_initializer()

# 启动图，开始执行
with tf.Session() as sess2:
    # 运行 初始化init 操作
    sess2.run(init_op)
    # 打印state 初始值
    print(sess2.run(state))
    # 执行 add op
    for _ in range(3):
        sess2.run(update)
        print(sess2.run(state))

print('======Fetch========' * 4)

'''
    Fetch:
       为了取回操作的输出内容, 
       可以在使用 Session 对象的 run() 调用 执行图时, 
       传入一些 tensor, 这些 tensor 会帮助你取回结果.  
在之前的例子里, 我们只取回了单个节点 state, 但是你也可以取回多个 tensor:
需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。
'''

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
add_op = tf.add(input2, input3)
mul_op = tf.multiply(input1, add_op)

with tf.Session() as sess:
    # 同时fetch mul，add 操作
    result = sess.run([mul_op, add_op])
    print(result)

print('======feed========' * 4)
'''
上述示例在计算图中引入了 tensor, 以常量或变量的形式存储. 
TensorFlow 还提供了 feed 机制, 
该机制可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁,
 直接插入一个 tensor.
 
 feed 使用一个 tensor 值临时替换一个操作的输出结果. 
 你可以提供 feed 数据作为 run() 调用的参数. 
  feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 
  最常见的用例是将某些特殊的操作指定为 "feed" 操作, 
  标记的方法是使用 tf.placeholder() 为这些操作创建占位符. 
'''
import numpy as np

input1 = tf.placeholder(np.float32)
input2 = tf.placeholder(np.float32)
mul_op = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run([mul_op], feed_dict={input1: [7.], input2: [2.]})
    print(result)
