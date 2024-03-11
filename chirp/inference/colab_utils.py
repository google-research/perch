# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for user-facing colab notebooks."""

import warnings

from absl import logging
from chirp import config_utils
from chirp.inference import embed_lib
import numpy as np
import tensorflow as tf


def initialize(use_tf_gpu: bool = True, disable_warnings: bool = True):
  """Apply notebook conveniences.

  Args:
    use_tf_gpu: If True, allows GPU use and sets Tensorflow to 'memory growth'
      mode (instead of reserving all available GPU memory at once). If False,
      Tensorflow is restricted to CPU operation. Must run before any TF
      computations to be effective.
    disable_warnings: If True, disables printed warnings from library code.
  """
  if disable_warnings:
    logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore')

  if not use_tf_gpu:
    tf.config.experimental.set_visible_devices([], 'GPU')
  else:
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)


def prstats(title: str, ar: np.ndarray):
  """Print summary statistics for an array."""
  tmpl = (
      '% 16s : \tshape: % 16s\tmin: %6.2f\tmean: %6.2f\tmax: %6.2f\tstd: %6.2f'
  )
  print(
      tmpl
      % (title, np.shape(ar), np.min(ar), np.mean(ar), np.max(ar), np.std(ar))
  )
