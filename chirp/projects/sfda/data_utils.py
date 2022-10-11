# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

"""Utilities to load data for Source-free Domain Adaptation."""
import ast
from typing import List, Tuple

from chirp.data import pipeline
from ml_collections import config_dict
import tensorflow as tf


def to_tf_compatible_split(split_tuple: List[Tuple[int, int]]) -> str:
  """Converts the splits specified in the config file into tf-readable format.

  TF dataset expects a split in the form 'train[x%:y%]'. The % sign
  can cause trouble when trying to iterate over splits (in a shell script for
  instance). As a quick workaround, at the config file level, we specify the
  splits using [(a, b), (x, y), ...], which will be converted to
  'train[a%:b%]+train[x%:y%]' in this function.

  Args:
    split_tuple: The train split, in the form [(a, b), (x, y), ...]

  Returns:
    The TF-readible version of split_tuple, i.e. 'train[a%:b%]+train[x%:y%]+...'
  """
  splits = []
  for st, end in split_tuple:
    splits.append(f"train[{st}%:{end}%]")
  return "+".join(splits)


def get_audio_datasets(
    adaptation_data_config: config_dict.ConfigDict,
    eval_data_config: config_dict.ConfigDict,
    sample_rate_hz: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Get datasets used for adaptation and evaluation.

  Args:
    adaptation_data_config: The configuration containing relevant information
      (e.g. transformation pipeline) to build the adaptation dataset.
    eval_data_config: The configuration containing relevant information to build
      the evaluation dataset.
    sample_rate_hz: The sample rate used by the current model. Used to
      double-check that this sample rate matches the one the data was created
      with.

  Returns:
    The datasets used for adaptation and evaluation.

  Raises:
    ValueError: If the model's sample_rate and data's sample_rate do not match.
  """
  adaptation_split = to_tf_compatible_split(
      ast.literal_eval(adaptation_data_config.split))
  eval_split = to_tf_compatible_split(ast.literal_eval(eval_data_config.split))

  # is_train only affects how data is processed by tensorflow internally,
  # in the case of a distributed setting. For now, SFDA is only supported in a
  # a non-distributed setting. Therefore, the is_train argument has no effect.
  adaptation_dataset, adaptation_dataset_info = pipeline.get_dataset(
      split=adaptation_split,
      is_train=False,
      dataset_directory=adaptation_data_config.dataset_directory,
      tfds_data_dir=adaptation_data_config.tfds_data_dir,
      pipeline=adaptation_data_config.pipeline,
  )

  if adaptation_dataset_info.features["audio"].sample_rate != sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            adaptation_dataset_info.features["audio"].sample_rate))

  # Grab the data used for evaluation
  val_dataset, val_dataset_info = pipeline.get_dataset(
      split=eval_split,
      is_train=False,
      dataset_directory=eval_data_config.dataset_directory,
      tfds_data_dir=eval_data_config.tfds_data_dir,
      pipeline=eval_data_config.pipeline,
  )

  if val_dataset_info.features["audio"].sample_rate != sample_rate_hz:
    raise ValueError(
        "Dataset sample rate must match config sample rate. To address this, "
        "need to set the sample rate in the config to {}.".format(
            val_dataset_info.features["audio"].sample_rate))
  return adaptation_dataset, val_dataset
