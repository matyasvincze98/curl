import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import pickle
import pandas as pd


patches_128k_dict = pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
patches_128k_train_dict = {key: patches_128k_dict[key] for key in ['train_images', 'train_labels']}
patches_128k_train_dict['train_images'] = patches_128k_train_dict['train_images'].reshape(-1, 48, 48, 1)
patches_128k_train_ds = tf.data.Dataset.from_tensor_slices(patches_128k_train_dict)
      
ds_train, ds_info = tfds.load(
      name='mnist',
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **{})

print(patches_128k_train_ds)
print(ds_train, ds_info)
