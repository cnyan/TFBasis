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
