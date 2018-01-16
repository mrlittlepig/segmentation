#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from net.resnet import inference_fn
import cv2
from PIL import Image

from tensorflow.python.platform import gfile

data_params = {
  'model_root': 'data',
  'data_dir': 'MNIST_data/mnist_test',
  'data_format': 'channels_last',
  'num_classes': 2,
  'image_size': [320, 240]
}

model_params = {
  'resnet': {
    'model_dir': os.path.join(
      data_params['model_root'],
      'resnet'
    ),
    'checkpoint_index': 16162
  }
}

init_params = {
  'data_params': data_params,
  'net_params': model_params['resnet'],
}

graph_params = {
  "source_checkpoint": os.path.join(
    init_params['net_params']['model_dir'],"model.ckpt-{}"
      .format(init_params['net_params']["checkpoint_index"])
  ),
  "inference_graph": os.path.join(
    init_params['net_params']["model_dir"],"inference_graph_{}.pb"
      .format(init_params['net_params']["checkpoint_index"])
  ),
  "input_nodes": "input",
  "output_nodes": "output"
}

def export_inference_graph():
  """
  Need inference graph fn as inference_fn,
  placeholder must feed by real numpy matrix.
  """
  output_file=graph_params["inference_graph"]

  with tf.Graph().as_default() as graph:
    placeholder = tf.placeholder(name=graph_params["input_nodes"], dtype=tf.float32,
                                shape=[None,init_params['data_params']['image_size'][0],
                                 init_params['data_params']['image_size'][1], 1])
    ones = np.ones([1, init_params['data_params']['image_size'][0],
                    init_params['data_params']['image_size'][1],1])

    print(ones.shape)

    sess = tf.Session()
    output = inference_fn(placeholder)
    sess.run(tf.global_variables_initializer())
    output_v = sess.run([output], feed_dict={placeholder:ones})
    graph_def = graph.as_graph_def()
    with gfile.GFile(output_file, 'wb') as f:
      f.write(graph_def.SerializeToString())


def predict(image_path):
  x = tf.placeholder(name=graph_params["input_nodes"], dtype=tf.float32,
                               shape=[None, data_params['image_size'][0],
                                      data_params['image_size'][1], 3])

  image = cv2.imread(image_path)

  image = np.reshape(image, [1, data_params['image_size'][0], data_params['image_size'][1], 3])
  image = np.asarray(image, dtype=np.float32)
  with tf.Session() as sess:
    y = inference_fn(image)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, graph_params["source_checkpoint"])
    print("Model restored.")
    result = sess.run(y, feed_dict={x: image})
    return result


if __name__ == "__main__":
  image_path = "dataset/tmp/0.png"
  result = predict(image_path)
  result = np.reshape(result, (320, 240, 2))
  result = result*255
  cv2.imshow("result", result[:,:,0])
  cv2.waitKey(10000)


