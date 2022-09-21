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

"""Utility functions for evaluation."""

from typing import Callable, Dict, Tuple

from chirp.data import pipeline
import ml_collections
import numpy as np
import tensorflow as tf

ConfigDict = ml_collections.ConfigDict


def _load_eval_dataset(dataset_config: ConfigDict) -> tf.data.Dataset:
  """Loads an evaluation dataset from its corresponding configuration dict."""
  return pipeline.get_dataset(
      split='train',
      is_train=False,
      dataset_directory=dataset_config.tfds_name,
      tfds_data_dir=dataset_config.tfds_data_dir,
      tf_data_service_address=None,
      pipeline=dataset_config.pipeline,
  )[0]


def load_eval_datasets(config: ConfigDict) -> Dict[str, tf.data.Dataset]:
  """Loads all evaluation datasets for a given evaluation configuration dict.

  Args:
    config: the evaluation configuration dict.

  Returns:
    A dict mapping dataset names to evaluation datasets.
  """
  return {
      dataset_name: _load_eval_dataset(dataset_config)
      for dataset_name, dataset_config in config.dataset_configs.items()
  }


def get_embeddings(
    dataset_name: str, dataset: tf.data.Dataset,
    model_callback: Callable[[np.ndarray], np.ndarray]
) -> Tuple[str, tf.data.Dataset]:
  """Embeds the audio slice in each tf.Example across the input dataset.

  Args:
    dataset_name: a string identifier of the dataset being operated on.
    dataset: a TF Dataset composed of tf.Examples.
    model_callback: a Callable that takes a NumPy array and produces an embedded
      NumPy array.

  Returns:
    A tuple of dataset_name and an updated TF Dataset with a new 'embedding'
    feature and deleted 'audio' feature.
  """

  def _map_func(example):
    example['embedding'] = tf.py_function(
        func=model_callback, inp=[example['audio']], Tout=tf.float32)
    del example['audio']
    return example

  # Use the 'audio' feature to produce a model embedding; delete the old 'audio'
  # feature.
  dataset = dataset.map(
      _map_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
  return dataset_name, dataset
