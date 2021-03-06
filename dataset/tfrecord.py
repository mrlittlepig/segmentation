# coding: utf-8

import os
import cv2
import tensorflow as tf
import numpy as np
import sys

sys.path.append("../")

from util import create_synthetic_imgs
#import util.create_synthetic_imgs as create_synthetic_imgs

HEIGHT = 320
WIDTH = 240

class tfrecord(object):

  def __init__(self, record_dir):
    self.root = record_dir
    self.train_dir = 'train'
    self.test_dir = 'test'
    self.num_trains = None
    self.num_tests = None
    self.train_images_list = []
    self.train_labels_list = []
    self.test_images_list = []
    self.test_labels_list = []
    self.__read_filelist()

  def __read_filelist(self):
    trainlist = os.path.join(self.root, 'train.txt')
    testlist = os.path.join(self.root, 'test.txt')
    assert os.path.exists(trainlist)
    assert os.path.exists(testlist)
    trainlines = file(trainlist, 'r').readlines()
    testlines = file(testlist, 'r').readlines()
    for line in trainlines:
      line  = line.strip()
      if len(line) == 9:
        self.train_images_list.append(os.path.join(self.root,self.train_dir, line))
      elif len(line) == 15:
        self.train_labels_list.append(os.path.join(self.root, self.train_dir, line))
      else:
        print "error format line: ", line
    assert len(self.train_images_list) == len(self.train_labels_list)
    self.num_trains = len(self.train_images_list)

    for line in testlines:
      line = line.strip()
      if len(line) == 9:
        self.test_images_list.append(os.path.join(self.root, self.test_dir, line))
      elif len(line) == 15:
        self.test_labels_list.append(os.path.join(self.root, self.test_dir, line))
      else:
        print "error format line: ", line
    assert len(self.test_images_list) == len(self.test_labels_list)
    self.num_tests = len(self.test_images_list)

  def create(self):
    train_file = os.path.join(self.root, 'train.tfrecord')
    test_file = os.path.join(self.root, 'test.tfrecord')
    train_writer = tf.python_io.TFRecordWriter(train_file)
    test_writer = tf.python_io.TFRecordWriter(test_file)
    for train_index in range(len(self.train_images_list)):
      print "processing %d..." % (train_index + 1)
      image = cv2.imread(self.train_images_list[train_index])
      image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
      images = create_synthetic_imgs.augment_image(image)

      label = cv2.imread(self.train_labels_list[train_index])
      label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
      label = cv2.resize(label, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

      label = create_synthetic_imgs.process_0_1(label)
      labels = create_synthetic_imgs.augment_image_label(label)
      for image_index in range(len(images)):
        image = np.array(images[image_index], dtype=np.uint8).tobytes()
        label = np.array(labels[image_index], dtype=np.uint8).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
          "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
          "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        }))
        train_writer.write(example.SerializeToString())
    train_writer.close()

    for test_index in range(len(self.test_images_list)):
      print "processing %d..." % (test_index + 1)
      image = cv2.imread(self.test_images_list[test_index])
      image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
      images = create_synthetic_imgs.augment_image(image)

      label = cv2.imread(self.test_labels_list[test_index])
      label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
      label = cv2.resize(label, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

      label = create_synthetic_imgs.process_0_1(label)
      labels = create_synthetic_imgs.augment_image_label(label)
      for image_index in range(len(images)):
        image = np.array(images[image_index], dtype=np.uint8).tobytes()
        label = np.array(labels[image_index], dtype=np.uint8).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
          "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
          "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        }))
        test_writer.write(example.SerializeToString())
    test_writer.close()

  def decode(self, record_path):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    filename_queue = tf.train.string_input_producer([record_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                         "label": tf.FixedLenFeature([], tf.string),
                                         "image": tf.FixedLenFeature([], tf.string)
                                       })

    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])

    label = tf.decode_raw(features["label"], tf.uint8)
    label = tf.reshape(label, [HEIGHT, WIDTH])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if not os.path.exists('tmp'):
      os.mkdir('tmp')
    for i in xrange(10):
      image_, label_ = sess.run([image, label])
      print(label_)
      cv2.namedWindow("b", cv2.WINDOW_NORMAL)
      cv2.imwrite("tmp/%d.png" % i, image_)
      cv2.imwrite('tmp/%d_label.png' % i, label_*255)
      cv2.imshow("b", image_)
      cv2.waitKey(1000)

    coord.request_stop()
    coord.join(threads)


def main():
  record = tfrecord('')
  record.create()
  # record.decode("train.tfrecord")


if __name__ == "__main__":
    main()
