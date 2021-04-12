from absl import app

import training

variance = 1.2
n_y = 30
lr_init = 1e-3
n_steps = 10000  # 100000
report_interval = 10
n_enc = [1200, 600, 300, 150]
n_dec = [500, 500]
n_z = 32
num_train = 50000  # 115200
num_test = 5000  # 12800


def main(unused_argv):
  training.run_training(
      dataset='mnist',
      output_type='gaussian',
      variance=variance,
      n_y=n_y,
      n_y_active=1,
      training_data_type='sequential',
      n_concurrent_classes=1,
      lr_init=lr_init,
      lr_factor=1.,
      lr_schedule=[1],
      blend_classes=False,
      train_supervised=False,
      n_steps=n_steps,
      report_interval=report_interval,
      knn_values=[10],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': n_enc,
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': n_dec,
          'dec_up_strides': None,
      },
      n_z=n_z,
      dynamic_expansion=False,
      ll_thresh=-200.0,
      classify_with_samples=False,
      gen_replay_type=None,
      use_supervised_replay=False,
      num_train=num_train,
      num_test=num_test,
      )

if __name__ == '__main__':
  app.run(main)
