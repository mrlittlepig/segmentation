from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import net.resnet as resnet

data_params = {
  'model_root': 'data',
  'data_dir': 'dataset',
  'data_format': 'channels_last',
  'num_classes': 2,
  'image_size': [320, 240, 3],
  'num_images': {
    'train': 20400,
    'validation': 3600
  }
}

net_params = {
  'lenet': {
    'net': 'lenet.model_fn',
    'model_dir': os.path.join(
      data_params['model_root'],
      'lenet'
    ),
  },
  'resnet': {
    'net': 'resnet.model_fn',
    'model_dir': os.path.join(
      data_params['model_root'],
      'resnet'
    ),
    'momentum': 0.9,
    'weight_decay': 2e-4,
    'resnet_size': 18
  }
}

init_params = {
  'batch_size': 120,
  'train_epoch': 100,
  'data_params': data_params,
  'net_params': net_params['resnet'],
}

model_fns = [resnet]
if not len(model_fns) == 0:
  model_fn = eval(init_params['net_params']['net'])

def input_fn(is_training, filename, batch_size=1, num_epochs=1):
  """A simple input_fn using the tf.data input pipeline."""

  def example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([init_params['data_params']['image_size'][0] *
                     init_params['data_params']['image_size'][1] *
                     init_params['data_params']['image_size'][2]])

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(features['label'], tf.uint8)
    label.set_shape([init_params['data_params']['image_size'][0] *
		     init_params['data_params']['image_size'][1]])
    # label = tf.cast(label, tf.uint8)
    return image, tf.one_hot(label, init_params['data_params']['num_classes'])

  dataset = tf.data.TFRecordDataset([filename])

  # Apply dataset transformations
  if is_training:
    dataset = dataset.shuffle(buffer_size=init_params['data_params']['num_images']['train'])

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Map example_parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(example_parser).prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels




def main():
  # Make sure that train and test data have been converted.
  train_file = os.path.join(init_params['data_params']['data_dir'], 'train.tfrecord')
  test_file = os.path.join(init_params['data_params']['data_dir'], 'test.tfrecord')
  assert (tf.gfile.Exists(train_file) and tf.gfile.Exists(test_file)), (
      'No TFRecord file exists.')

  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=init_params['net_params']['model_dir'],
      params={'data_format': init_params['data_params']['data_format'],
              'num_classes': init_params['data_params']['num_classes'],
              'batch_size': init_params['batch_size'],
              'net_params': init_params['net_params'],
              'num_images': init_params['data_params']['num_images']
              })

  # Set up train hook that logs the train accuracy every 100 steps.
  tensors_to_log = {
      'train_accuracy': 'train_accuracy'
  }
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  classifier.train(
      input_fn=lambda: input_fn(
          True, train_file, init_params['batch_size'], init_params['train_epoch']),
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_results = classifier.evaluate(
      input_fn=lambda: input_fn(False, test_file, init_params['batch_size']))
  print()
  print('Evaluation results:\n\t%s' % eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  main()
