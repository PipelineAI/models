from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import os

def transform(saved_model_path):
  optimized_model_path = '%s/optimized' % saved_model_path

  os.makedirs(optimized_model_path, exist_ok=True)

  converter = tf.contrib.lite.TocoConverter.from_saved_model(saved_model_path)
  tflite_model = converter.convert()

  file_size_bytes = open('%s/optimized_model.tflite' % optimized_model_path, "wb").write(tflite_model)
  print('Model File Size: %s' % file_size_bytes)

def main(args):
  transform(args[1])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  absl_app.run(main)
