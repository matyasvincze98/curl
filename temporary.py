from random import shuffle
import glob
import sys
import cv2
import numpy as np
#import skimage.io as io
import tensorflow as tf

ds_train_, ds_info_ = tfds.load(
      name='mnist',
      split=tfds.Split.TRAIN,
      with_info=True,
      as_dataset_kwargs={'shuffle_files': False},
      **{})

print(type(ds_train_), ds_train_)
print()
print(type(ds_info_), ds_info_)


'''
import pickle
import pandas as pd
patches_128k = pd.read_pickle('https://wigner.hu/~fcsikor/textures/labeled_texture_oatleathersoilcarpetbubbles_subsamp1_filtered_128000_48px.pkl')
'''
