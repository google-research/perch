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
from typing import Any
from absl import logging
from chirp.data import pipeline
from chirp.projects.sfda import models
import jax
from ml_collections import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds


def to_tf_compatible_split(split_tuple: list[tuple[int, int]]) -> str:
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
    splits.append(f'train[{st}%:{end}%]')
  return '+'.join(splits)


def get_audio_datasets(
    adaptation_data_config: config_dict.ConfigDict,
    eval_data_config: config_dict.ConfigDict,
    sample_rate_hz: float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  """Get audio datasets used for adaptation and evaluation.

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

  if adaptation_dataset_info.features['audio'].sample_rate != sample_rate_hz:
    raise ValueError(
        'Dataset sample rate must match config sample rate. To address this, '
        'need to set the sample rate in the config to {}.'.format(
            adaptation_dataset_info.features['audio'].sample_rate))

  # Grab the data used for evaluation
  val_dataset, val_dataset_info = pipeline.get_dataset(
      split=eval_split,
      is_train=False,
      dataset_directory=eval_data_config.dataset_directory,
      tfds_data_dir=eval_data_config.tfds_data_dir,
      pipeline=eval_data_config.pipeline,
  )

  if val_dataset_info.features['audio'].sample_rate != sample_rate_hz:
    raise ValueError(
        'Dataset sample rate must match config sample rate. To address this, '
        'need to set the sample rate in the config to {}.'.format(
            val_dataset_info.features['audio'].sample_rate))
  return adaptation_dataset, val_dataset


def get_image_datasets(
    image_model: models.ImageModelName,
    dataset_name: str,
    batch_size_train: int,
    batch_size_eval: int,
    data_seed: int,
    builder_kwargs: dict[str, Any],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  """Get image dataset used for adaptation and evaluation.

  Args:
    image_model: The image model used for adaptation. This dictates the input
      pipeline to use.
    dataset_name: The name of the dataset used for adaptation and evaluation.
    batch_size_train: The batch size used for adaptation.
    batch_size_eval: The batch size used for evaluation.
    data_seed: Used to seed data shuffling.
    builder_kwargs: Kwargs to pass when creating the data builder.

  Returns:
    The adaptation and evaluation datasets.
  """
  input_pipeline = models.MODEL_REGISTRY[image_model](
      num_classes=0).get_input_pipeline
  dataset_metadata = get_metadata(dataset_name)
  num_devices = jax.local_device_count()

  def build_image_dataset(split: str, batch_size: int):
    data_builder = tfds.builder(dataset_name, **builder_kwargs)
    tfds_split = dataset_metadata['splits'][split]
    logging.info('Using split %s for dataset %s', tfds_split, dataset_name)
    dataset = input_pipeline(
        data_builder=data_builder,
        split=tfds_split,
        image_size=dataset_metadata['resolution'])
    if split == 'train':
      dataset = dataset.shuffle(512, seed=data_seed)
    if num_devices is not None:
      dataset = dataset.batch(batch_size // num_devices, drop_remainder=False)
      dataset = dataset.batch(num_devices, drop_remainder=False)
    else:
      dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(10)

  adaptation_dataset = build_image_dataset('train', batch_size_train)
  val_dataset = build_image_dataset('eval', batch_size_eval)
  return adaptation_dataset, val_dataset


def get_metadata(dataset_name: str) -> dict[str, Any]:
  """Maps image dataset names to metadata.

  Args:
    dataset_name: The raw dataset_name.

  Returns:
    A dictionary of metadata for this dataset, including:
      - num_classes: The number of classes.
      - resolution: The image resolution.

  Raises:
    NotImplementedError: If the dataset is unknown.
  """
  if 'imagenet' in dataset_name:
    if 'corrupted' in dataset_name:
      split = {'train': 'validation[:75%]', 'eval': 'validation[75%:]'}
    else:
      split = {'train': 'test[:75%]', 'eval': 'test[75%:]'}
    return {'num_classes': 1000, 'resolution': 224, 'splits': split}
  elif 'cifar' in dataset_name:
    split = {'train': 'test[:75%]', 'eval': 'test[75%:]'}
    return {'num_classes': 10, 'resolution': 32, 'splits': split}
  elif dataset_name == 'fake_image_dataset':
    split = {'train': 'train[:1]', 'eval': 'train[1:2]'}
    return {'num_classes': 2, 'resolution': 12, 'splits': split}
  elif dataset_name == 'vis_da_c':
    # In line with NRC's results, we do both the adaptation and the evaluation
    # on the validation set of VisDA-C.
    split = {'train': 'validation[:75%]', 'eval': 'validation[75%:]'}
    return {'num_classes': 12, 'resolution': 224, 'splits': split}
  else:
    raise NotImplementedError(
        f'Unknown number of classes for dataset {dataset_name}.')
