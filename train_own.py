from absl import app

import argparse
import os

import training

parser = argparse.ArgumentParser()
parser.add_argument("--variance", default=0.2, type=float)
parser.add_argument("--n_y", default=30, type=int)
parser.add_argument("--lr_init", default=1e-3, type=float)
parser.add_argument("--n_steps", default=100000, type=int)
parser.add_argument("--report_interval", default=10000, type=int)
parser.add_argument("--n_enc", default=[1200, 600, 300, 150], type=list)
parser.add_argument("--n_dec", default=[500, 500], type=list)
parser.add_argument("--n_z", default=32, type=int)
parser.add_argument("--num_train", default=115200, type=int)
parser.add_argument("--num_test", default=12800, type=int)
args = parser.parse_args([] if "__file__" not in globals() else None)


def main(unused_argv):
  training.run_training(
      dataset='mnist',
      output_type='gaussian',
      variance=args.variance,
      n_y=args.n_y,
      n_y_active=1,
      training_data_type='sequential',
      n_concurrent_classes=1,
      lr_init=args.lr_init,
      lr_factor=1.,
      lr_schedule=[1],
      blend_classes=False,
      train_supervised=False,
      n_steps=args.n_steps,
      report_interval=args.report_interval,
      knn_values=[10],
      random_seed=1,
      encoder_kwargs={
          'encoder_type': 'multi',
          'n_enc': args.n_enc,
          'enc_strides': [1],
      },
      decoder_kwargs={
          'decoder_type': 'single',
          'n_dec': args.n_dec,
          'dec_up_strides': None,
      },
      n_z=args.n_z,
      dynamic_expansion=False,
      ll_thresh=-200.0,
      classify_with_samples=False,
      gen_replay_type=None,
      use_supervised_replay=False,
      num_train=args.num_train,
      num_test=args.num_test,
      )

if __name__ == '__main__':
  app.run(main)
