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

"""A template for any image model."""

from typing import Optional

from chirp.models import output
from etils import epath
import flax
import flax.linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds


class ImageModel(nn.Module):
  """A template for any image model."""

  @nn.compact
  def __call__(
      self, x, train: bool, use_running_average: Optional[bool]
  ) -> output.ClassifierOutput:
    """Just like any standard nn.Module, defines the foward pass of the model.

    We formulate two non-standard requirements for the forward pass. First, it
    must disentangle the train/test behavior of BatchNorm layers and those of
    other noisy layers (e.g. Dropout). This is achieved through the use of
    'train' and 'use_running_average' options. Second, we require that the
    outputs are packaged into a output.ClassifierOutput, thereby including
    both the encoder's features, as well as the head's output. See
    chirp/projects/sfda/models/resnet.py for an example.

    Args:
      x: Batch of input images.
      train: Whether this is training. This affects noisy layers' behavior (e.g.
        Dropout). It also affects BatchNorm behavior in case
        'use_running_average' is set to None.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.

    Returns:
      The model's outputs, packaged as a output.ClassifierOutput.
    """
    raise NotImplementedError

  @staticmethod
  def load_ckpt(dataset_name: str) -> flax.core.frozen_dict.FrozenDict:
    """Loads the checkpoint for the current dataset.

    Args:
      dataset_name: The current dataset used.

    Returns:
      variables: The flax variables corresponding to the loaded checkpoint.
    """
    raise NotImplementedError

  @staticmethod
  def get_ckpt_path(dataset_name: str) -> epath.Path:
    """Returns the path to the checkpoint for the current dataset.

    Using a separate function from 'load_ckpt' (the latter uses 'get_ckpt_path')
    to make it easier to verify checkpoints paths.

    Args:
      dataset_name: The current dataset used.

    Returns:
      variables: The path to load the checkpoint.
    """
    raise NotImplementedError

  @staticmethod
  def is_bn_parameter(parameter_name: list[str]) -> bool:
    """Verifies whether some parameter belong to a BatchNorm layer.

    Args:
      parameter_name: The name of the parameter, as a list in which each member
        describes the name of a layer. E.g. ('Block1', 'batch_norm_1', 'bias').

    Returns:
      True if this parameter belongs to a BatchNorm layer.
    """
    raise NotImplementedError

  @staticmethod
  def get_input_pipeline(
      data_builder: tfds.core.DatasetBuilder, split: str, **kwargs
  ) -> tf.data.Dataset:
    """Get the data pipeline for the current model.

    Because we're relying on pretrained models from the web, this part of the
    data pipeline can hardly be factorized. We hereby provide a default
    pipeline that converts image to tf.float32, and one-hots the labels.
    However, we **leave it for each model to specify its own processing
    pipeline**, with the only requirement of producing one-hot labels.

    Args:
      data_builder: The dataset's data builder.
      split: The split of the dataset used.
      **kwargs: Additional kwargs that may be useful for model-specific
        pipelines.

    Returns:
      The processed dataset.
    """
    read_config = tfds.ReadConfig(add_tfds_id=True)
    dataset = data_builder.as_dataset(split=split, read_config=read_config)

    def _pp(example):
      image = tf.image.convert_image_dtype(example['image'], tf.float32)
      label = tf.one_hot(
          example['label'], data_builder.info.features['label'].num_classes
      )

      return {'image': image, 'label': label, 'tfds_id': example['tfds_id']}

    return dataset.map(_pp, tf.data.experimental.AUTOTUNE)
