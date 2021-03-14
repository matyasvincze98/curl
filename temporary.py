import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import pandas as pd

ds_train, ds_info = tfds.load(
      name='mnist',
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **{})

ds_numpy = tfds.as_numpy(ds_train)
for ex in ds_numpy:
      print(ex)
      break

patches_128k_dict = pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
print(patches_128k_dict['train_images'][0], patches_128k_dict['train_labels'][0])

'''
patches_128k_tf = tf.data.Dataset.from_tensor_slices(patches_128k_pd)
'''
