import tensorflow as tf
import pickle
import pandas as pd

patches_128k = tf.data.Dataset.from_tensor_slices(
      pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
)
print(patches_128k)

'''
ds_train_, ds_info_ = tfds.load(
      name='mnist',
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **{})
'''
