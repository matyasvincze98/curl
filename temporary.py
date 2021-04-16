import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import pickle
import pandas as pd
     
import training
batch_size = 100
test_batch_size = 1000
dataset_kwargs = {}
image_key = 'train_images'
label_key = 'train_labels'
dataset_ops = training.get_data_sources('mnist', dataset_kwargs, batch_size,
                                 test_batch_size, 'iid',
                                 1, image_key, label_key, 100, 10)
print(dataset_ops.train_data)
print(dataset_ops.train_data_for_clf)
print(dataset_ops.test_data)

with tf.Session() as sess:
      print(sess.run(dataset_ops.test_data['test_images']))
      print(sess.run(dataset_ops.test_data['test_labels']))
