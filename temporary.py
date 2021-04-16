import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import pickle
import pandas as pd

patches_128k_dict = pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
patches_128k_train_dict = {key: patches_128k_dict[key] for key in ['train_images', 'train_labels']}
patches_128k_train_dict['train_images'] = patches_128k_train_dict['train_images'][:10].reshape(-1, 48, 48, 1)
patches_128k_train_dict['train_labels'] = patches_128k_train_dict['train_labels'][:10]
patches_128k_test_dict = {key: patches_128k_dict[key] for key in ['test_images', 'test_labels']}
patches_128k_test_dict['test_images'] = patches_128k_test_dict['test_images'][:10].reshape(-1, 48, 48, 1)
patches_128k_test_dict['test_labels'] = patches_128k_test_dict['test_labels'][:10]
train_ds = tf.data.Dataset.from_tensor_slices(patches_128k_train_dict)
test_ds = tf.data.Dataset.from_tensor_slices(patches_128k_test_dict)

ds_train, ds_info = tfds.load(
      name='mnist',
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **{})
      
'''
import training
batch_size = 100
test_batch_size = 1000
dataset_kwargs = {}
image_key = 'train_images'
label_key = 'train_labels'
dataset_ops = training.get_data_sources('mnist', dataset_kwargs, batch_size,
                                 test_batch_size, 'sequential',
                                 1, image_key, label_key)
print(dataset_ops.train_data)
print(dataset_ops.train_data_for_clf)
print(dataset_ops.test_data)
'''

q = train_ds.make_one_shot_iterator().get_next()
qq = ds_train.make_one_shot_iterator().get_next()
with tf.Session() as sess:
      print(sess.run(q))
      print(sess.run(qq))
