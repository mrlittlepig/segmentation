#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

imageSZ = {'rows': 320, 'cols': 240, 'channels': 3}

# 制作二进制数据
def recordsCreater(label_file, dst_records):
    writer = tf.python_io.TFRecordWriter(dst_records)

    lines = readLines(label_file)
    label_record = map(lines)
    file_name_length = len(file.getFileName(label_file))
    images_dir = label_file[:-1 * file_name_length]
    index = 0
    for name in label_record:
        index = index + 1
        image_file = images_dir + str(label_record[name]) + '/' + name
        img = Image.open(image_file)
        img = img.resize((imageSZ['rows'], imageSZ['cols']))
        bytesImg = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_record[name])])),
                'bytesImg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytesImg]))
            }))
        if index % 100 == 0:
            print("processing %d:" % index + images_dir + name)
        writer.write(example.SerializeToString())
    print("done!")
    writer.close()

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def labels_one_hot(labels, num_classes):
    label_XS = np.empty(len(labels))
    for i in range(len(labels)):
        label_XS[i] = np.int(labels[i])
    return dense_to_one_hot(np.array(label_XS, dtype=np.uint8), num_classes)

def images_modifier(images, labels, batch_size=100):
    return images.reshape(batch_size, 28*28*1), labels_one_hot(labels, 10)

# 读取二进制数据
def recordsReader(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'bytesImg': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['bytesImg'], tf.uint8)
    image = tf.reshape(image, [imageSZ['rows'],imageSZ['cols']])
    label = tf.cast(features['label'], tf.int32)
    return image, label

def test_reader(recordsFile):
    image, label = recordsReader(recordsFile)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            image, label = sess.run([image, label])  # 在会话中取出image和label
            # img = Image.fromarray(example, 'RGB')  # 如果img是RGB图像
            # img = Image.fromarray(example)
            #
            # img.save('./' + '_'+'Label_' + str(l) + '.jpg')  # 存下图片
            Image._show(Image.fromarray(image))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    test_label_file, test_dst_records = "../MNIST_data/mnist_test/test.txt", "../MNIST_data/mnist_test.tfrecords"
    # train_label_file, train_dst_records = "../MNIST_data/mnist_train/train.txt", "../MNIST_data/mnist_train.tfrecords"
    # recordsCreater(test_label_file, test_dst_records)
    # recordsCreater(train_label_file, train_dst_records)
    test_reader(test_dst_records)
    # print(dense_to_one_hot(1, 10))
