"""Training file to run most of the experiments in the paper.

The default parameters corresponding to the first set of experiments in Section
4.2.

For the expansion ablation, run with different ll_thresh values as in the paper.
Note that n_y_active represents the number of *active* components at the
start, and should be set to 1, while n_y represents the maximum number of
components allowed, and should be set sufficiently high (eg. n_y = 100).

For the MGR ablation, setting use_sup_replay = True switches to using SMGR,
and the gen_replay_type flag can switch between fixed and dynamic replay. The
generative snapshot period is set automatically in the train_curl.py file based
on these settings (ie. the data_period variable), so the 0.1T runs can be
reproduced by dividing this value by 10.
"""

from absl import app
from absl import flags

import training


def main(unused_argv):
  training.run_training(
      dataset='mnist',
      output_type='bernoulli',
      n_y=30,
      n_y_active=1,
      training_data_type='sequential',
      n_concurrent_classes=1,
      lr_init=1e-3,
      lr_factor=1.,
      lr_schedule=[1],
      blend_classes=False,
      train_supervised=False,
      n_steps=100000,
      report_interval=10000,
      knn_values=[10],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': [1200, 600, 300, 150],
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': [500, 500],
          'dec_up_strides': None,
      },
      n_z=32,
      dynamic_expansion=False,
      ll_thresh=-200.0,
      classify_with_samples=False,
      gen_replay_type=None,
      use_supervised_replay=False,
      )

if __name__ == '__main__':
  app.run(main)
