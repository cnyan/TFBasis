# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/7/26 16:47
@Describe：
    读取数据并对其进行数据增强处理
    数据集下载链接：http://www.cs.toronto.edu/~kriz/cifar.html
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_classes = 10
# 定义用于训练和评估的样本总数
num_example_pre_epoch_for_train = 50000
num_example_pre_epoch_for_eval = 10000


# 定义一个类，用于返回读取的Cifar-10数据
class CIFAR10Record(object):
    pass


# 定义读取 Cifar10数据的函数
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3  # RGB通道

    image_bytes = result.height * result.width * result.depth  # 数据长度= 3072

    # 每个样本都包含一个label数据和image数据，结果为record_bytes=3073
    record_bytes = label_bytes + image_bytes  # 3073

    # 创建一个文件读取类，并调用该类的read()函数从文件队列中读取文件
    # FixedLengthRecordReader类用于读取（一行）固定长度字节数信息
    reader = tf.compat.v1.FixedLengthRecordReader(record_bytes=record_bytes)
    # 阅读器的read方法会输出一个key来表征输入的文件和其中的纪录(对于调试非常有用)
    # 同时得到一个字符串标量，这个字符串标量可以被一个或多个解析器，或者转换操作将其解码为张量并且构造成为样本。
    result.key, value = reader.read(file_queue)

    # 得到的value就是record_bytes 长度的包含多个lebel数据和image数据的字符串
    # decode_raw() 函数可以将字符串解析成图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.compat.v1.uint8)

    # 将得到的record_bytes数组中的第一个元素类型转换为int32类型
    # strided_slice()函数用于对input截取从[begin, end)区间的数据
    result.label = tf.compat.v1.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.compat.v1.int32)
    # 剪切label之后剩下的就是图片数据,我们将这些数据的格式从[depth * height * width]
    # 转为[depth, height, width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    # 将[depth, height, width]的格式转变为[height, width, depth]的格式
    # transpose()函数用于原型为     transpose(x,perm,name)
    result.uint8image = tf.compat.v1.transpose(depth_major, [1, 2, 0])
    return result


def inputs(data_dir, batch_size, distorted):
    # 使用os的join()函数拼接路径
    filenames = [os.path.join(data_dir, "src/data_batch_%d.bin" % i) for i in range(1, 6)]

    # 创建一个文件队列，并调用read_cifar10()函数读取队列中的文件
    file_queue = tf.compat.v1.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)

    # 使用cast()函数将图片数据转为float32格式，原型cast(x,DstT,name)
    reshaped_image = tf.compat.v1.cast(read_input.uint8image, tf.compat.v1.float32)
    num_example_pre_epoch = num_example_pre_epoch_for_train

    # 对图像数据进行数据增强处理:翻转、随机剪切
    if distorted != None:
        # 将[32,32,3]大小的图片随机裁剪成[24,24,3]大小
        cropped_image = tf.compat.v1.random_crop(reshaped_image, [24, 24, 3])
        # 随机左右翻转图片
        flipped_image = tf.compat.v1.image.flip_left_right(cropped_image)
        # 使用random_brightness()函数调整亮度
        # 函数原型random_brightness(image,max_delta,seed)
        adjusted_brightness = tf.compat.v1.image.random_brightness(flipped_image, max_delta=0.8)
        # 调整对比度
        adjusted_contrast = tf.compat.v1.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        # 标准化图片，注意不是归一化
        # per_image_standardization()是对每一个像素减去平均值并除以像素方差
        float_image = tf.compat.v1.image.per_image_standardization(adjusted_contrast)

        # 设置图片数据及label的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_example_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # 使用shuffle_batch()函数随机产生一个batch的image和label
        # 函数原型shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue,
        #      num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)
        images_train, labels_train = tf.compat.v1.train.shuffle_batch([float_image, read_input.label],
                                                                      batch_size=batch_size, num_threads=16,
                                                                      capacity=min_queue_examples + 3 * batch_size,
                                                                      min_after_dequeue=min_queue_examples)
        return images_train, tf.compat.v1.reshape(labels_train, [batch_size])
    # 不对图像数据进行数据增强处理
    else:
        # 随机裁剪，深度不变3
        resized_image = tf.compat.v1.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 没有图像的其他处理过程，直接标准化
        float_image = tf.compat.v1.image.per_image_standardization(resized_image)

        # 设置图片数据及label的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_example_pre_epoch * 0.4)
        # 使用batch()函数创建样例的batch，这个过程使用最多的是shuffle_batch()函数
        # 但是这里使用batch()函数代替了shuffle_batch()函数
        images_test, labels_test = tf.compat.v1.train.batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.compat.v1.reshape(labels_test, [batch_size])
