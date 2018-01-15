#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

image_size = (320, 240, 3)
channel = 3
num_classes = 2


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)

def protrait_fcn_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):

  if data_format is None:
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')


  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, 'input')
    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'input_batch_norm')

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=48, kernel_size=5, strides=4,
      data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.identity(inputs, 'initial_batch_norm')
    block1 = block_layer(
      inputs=inputs, filters=48, block_fn=block_fn, blocks=layers[0],
      strides=1, is_training=is_training, name='block_layer1',
      data_format=data_format)

    branch1 = conv2d_fixed_padding(block1, filters=8, kernel_size=1, strides=1,
                                   data_format=data_format)
    branch1 = tf.nn.relu(branch1, name='branch1_relu')


    block2 = block_layer(
      inputs=block1, filters=64, block_fn=block_fn, blocks=layers[1],
      strides=2, is_training=is_training, name='block_layer2',
      data_format=data_format)

    branch2 = conv2d_fixed_padding(block2, filters=8, kernel_size=1, strides=1,
                                   data_format=data_format)

    branch2 = tf.nn.relu(branch2, name='branch2_relu')
    block3 = block_layer(
      inputs=block2, filters=96, block_fn=block_fn, blocks=layers[2],
      strides=2, is_training=is_training, name='block_layer3',
      data_format=data_format)

    block4 = block_layer(
      inputs=block3, filters=8, block_fn=block_fn, blocks=1,
      strides=1, is_training=is_training, name='block_layer4',
      data_format=data_format)
    batch_norm_relu_block4 = batch_norm_relu(block4, is_training, data_format)
    upscore1 = tf.layers.conv2d_transpose(batch_norm_relu_block4, filters=8,kernel_size=4,
                                          strides=2, padding='same', name='conv_transpose1')
    fuse1 = tf.add(upscore1, branch2)

    upscore2 = tf.layers.conv2d_transpose(fuse1, filters=8, kernel_size=4,
                                          strides=2, padding='same', name='conv_transpose2')
    fuse2 = tf.add(upscore2, branch1)

    fuse2 = conv2d_fixed_padding(fuse2, filters=4, kernel_size=1, strides=1,
                                   data_format=data_format)
    upscore3 = tf.layers.conv2d_transpose(fuse2, filters=4, kernel_size=8,
                                          strides=4, padding='same', name='conv_transpose3')
    upscore3 = conv2d_fixed_padding(upscore3, filters=2, kernel_size=1, strides=1,
                                    data_format=data_format)

    outputs = tf.identity(upscore3, 'final_upscore')

    outputs = tf.reshape(outputs, (-1, 240*320 ,2))
    return outputs


  return model

def portrait_fcn_resnet_v2(resnet_size, num_classes, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return protrait_fcn_resnet_v2_generator(
      params['block'], params['layers'], num_classes, data_format)

def model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  features = tf.reshape(features, [-1, image_size[0], image_size[1], image_size[2]])
  tf.summary.image('images', features, max_outputs=6)


  network = portrait_fcn_resnet_v2(
      params['net_params']['resnet_size'], params['num_classes'], params['data_format'])
  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + params['net_params']['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 256, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 256
    batches_per_epoch = params['num_images']['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [10, 20, 30, 40]]
    values = [
        initial_learning_rate * decay for decay in [0.1, 0.02, 0.004, 0.0008, 0.000125]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    #tf.train.exponential_decay()
    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['net_params']['momentum'])

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])

  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

def inference_fn(images):
  """Our model_fn for ResNet to be used with our Estimator."""

  network = portrait_fcn_resnet_v2(resnet_size=18, num_classes=2)
  logits = network(inputs=images, is_training=False)
  output = tf.nn.softmax(logits, name='softmax')
  return output


