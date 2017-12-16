#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

image_size = (28, 28)
channel = 1
num_classes = 10

def mnist_model(inputs, mode, data_format):
  """Takes the MNIST inputs and mode and outputs a tensor of logits."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  inputs = tf.reshape(inputs, [-1, image_size[0], image_size[1], 1])

  if data_format is None:
    # When running on GPU, transpose the data from channels_last (NHWC) to
    # channels_first (NCHW) to improve performance.
    # See https://www.tensorflow.org/performance/performance_guide#data_formats
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else
                   'channels_last')

  if data_format == 'channels_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Logits layer
  logits = tf.layers.dense(inputs=dropout, units=num_classes)
  return logits

def model_fn(features, labels, mode, params):
  """Model function for MNIST."""
  logits = mnist_model(features, mode, params['data_format'])

  predictions = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  # Configure the training op
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)