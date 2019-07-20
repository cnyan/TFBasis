# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/20 11:04
@Describe：

"""
from actionRecognition1 import input_data

# 数据处理

actions_data = input_data.read_data_sets('src/actions_data.npy')
print(actions_data['all_data'][1])
print(actions_data['all_labels'][:20])
print(actions_data['validation_data'])
print(actions_data['validation_label'])

