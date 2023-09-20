# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Parts of the audio data pipeline that require customization."""
import dataclasses

from absl import logging
from chirp.preprocessing import pipeline
import jax
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class Batch(pipeline.DatasetPreprocessOp):
  """Collects samples into batches.

  The original Batch operation in chirp/data/pipeline.py drops the remainder
  by default. Combined with shuffling, this results in two runs over the dataset
  resulting in a potentially disjoint set of samples. Some methods, e.g. NOTELA
  or SHOT, assign pseudo-labels to samples before an epoch starts, and try to
  have the model match those pseudo-labels on-the-go. Therefore, those methods
  require a consistent set of samples across runs over the dataset. We solve
  this by simply keeping the remainder, thereby ensuring that all samples are
  seen at every run.

  Attributes:
    batch_size: The batch size to use.
    split_across_devices: If true, the minibatch will be split into smaller
      minibatches to be distributed across the local devices present. This is
      useful for distributed training.
  """

  batch_size: int
  split_across_devices: bool = False

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    if self.split_across_devices:
      if self.batch_size % jax.device_count():
        raise ValueError(
            f'batch size ({self.batch_size}) must be divisible by '
            f'number of devices ({jax.device_count()}).'
        )
      logging.info(
          'Splitting batch across %d devices, with local device count %d.',
          jax.device_count(),
          jax.local_device_count(),
      )
      dataset = dataset.batch(
          self.batch_size // jax.device_count(), drop_remainder=False
      )
      return dataset.batch(jax.local_device_count(), drop_remainder=False)
    else:
      return dataset.batch(self.batch_size, drop_remainder=False)
