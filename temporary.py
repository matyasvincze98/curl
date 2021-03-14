import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import pandas as pd


patches_128k_dict = pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
patches_128k_train_dict = {key: patches_128k_dict[key] for key in ['train_images', 'train_labels']}
print(patches_128k_train_dict)
patches_128k_train_df = pd.DataFrame.from_dict(patches_128k_train_dict['train_images'])
print(patches_128k_train_df.head())

# patches_128k_tf = tf.data.Dataset.from_tensor_slices(patches_128k_pd)

'''
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
'''
