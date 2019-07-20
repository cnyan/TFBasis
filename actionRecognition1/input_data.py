# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/20 11:06
@Describe：
    数据准备
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os

# 文件路径
ACTION_ROOT_PATH = 'D:\\temp\\actionData'


# ER 表示开头序号
def save_data_sets(SHOW_PLOT=False):
    '''
    :param beginActionModel: 动作开始的判断方法，regression 或 slope
    :param SHOW_PLOT:
    :return:
    '''
    # 导入数据 8 种动作，每种动作由7个节点数据构成
    file_list = []
    for maindir, subdir, file_name_list in os.walk(ACTION_ROOT_PATH):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
    # print(file_list)

    action_array = np.zeros(shape=(0, 1900))
    for filename in file_list:
        datamat = []
        labels = []

        lab = int(filename.split('\\')[-1].split('.')[0]) - 1
        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label[lab] = 1

        df = np.array(pd.read_csv(filename, dtype=float)).round(6)[:, 1:]
        df = df[:int(len(df) / 30) * 30, :]

        data = np.reshape(df, (-1, 30, 63))
        # print(data.shape)
        for i in range(len(data)):
            datamat.append(data[i].flatten())
            labels.append(label)
        # print(np.array(datamat).shape)
        # print(np.array(labels).shape)
        complex_array = np.concatenate([np.array(datamat), np.array(labels)], axis=1)
        # print(complex_array.shape)
        action_array = np.concatenate([action_array, complex_array], axis=0)
        # action_dict = {'data': np.array(datamat), 'labels': np.array(labels)}
        # print(action_dict['data'])
        # print('======' * 5)

    # print(action_array.shape)
    print(action_array[1, :-11])
    print(action_array[1, -10:])
    # 随机排序
    np.random.shuffle(action_array)
    print(action_array[1, :-11])
    print(action_array[1, -10:])

    # action_data = {'data': action_array[:, :-11], 'labels': action_array[:, -10:], 'original_data': action_array}
    # print(action_data['data'])
    # print(action_data['labels'])
    np.save('src/actions_data', action_array)


def read_data_sets(file_path='src/actions_data.npy', test_size=0.3):
    '''
    :param file_path: 需要读取的文件路径
    :param test_size: 随机数，用来区分训练集和测试机
    :return:
    '''
    df_array = np.load(file_path)
    np.random.shuffle(df_array)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_array[:, :-11], df_array[:, -10:], test_size=test_size)

    np.random.shuffle(df_array)
    validation = df_array[:int(len(df_array) * test_size)]
    X_validation, y_validation = validation[:, :-11], validation[:, -10:]

    action_data = {'all_data': df_array[:, :-11], 'all_labels': df_array[:, -10:], 'train_data': X_train,
                   'test_data': X_test, 'train_label': y_train, 'test_label': y_test, 'validation_data': X_validation,
                   'validation_label': y_validation, 'original_data': df_array}
    return action_data


if __name__ == '__main__':
    # save_data_sets()
    read_data_sets()
