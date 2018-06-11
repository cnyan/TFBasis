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

    使用图 (graph) 来表示计算任务.
    在被称之为 会话 (Session) 的上下文 (context) 中执行图.
    使用 tensor 表示数据.
    通过 变量 (Variable) 维护状态.
    使用 feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print('创建一个常量 V1 ，它是一个1行2列的矩阵')
v1 = tf.constant([[2, 3]])
print(v1)

print('创建一个常量 V2，它是一个2行1列的矩阵')
v2 = tf.constant([[2], [3]])
print(v2)

print('定义一个矩阵乘法，在会话中才会执行')
product = tf.matmul(v1, v2)

print('打印 product 得到的不是乘法之后的结果，而是乘法本身')
print(product)

# 定义一个会话
sess = tf.Session()
# 运算乘法，得到一个结果
result = sess.run(product)
print('打印会话结果')
print(result)
# 关闭会话
sess.close()
