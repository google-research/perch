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

"""Chirp custom TFDS Features."""


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Int16AsFloatTensor(tfds.features.Audio):
  """An int16 tfds.features.Tensor represented as a float32 in [-1, 1).

  Examples are stored as int16 tensors but encoded from and decoded into float32
  tensors in the [-1, 1) range (1 is excluded because we divide the
  [-2**15, 2**15 - 1] interval by 2**15).
  """

  INT16_SCALE = float(1 << 15)
  ALIASES = ['chirp.data.bird_taxonomy.bird_taxonomy.Int16AsFloatTensor']

  def __init__(
      self,
      *,
      file_format: str | None = None,
      shape: tfds.typing.Shape,
      dtype: tf.dtypes.DType = tf.float32,
      sample_rate: tfds.typing.Dim,
      encoding: str | tfds.features.Encoding = tfds.features.Encoding.NONE,
      doc: tfds.features.DocArg = None,
      lazy_decode: bool = False,
  ):
    del file_format
    del dtype

    self._int16_tensor_feature = tfds.features.Tensor(
        shape=shape, dtype=tf.int16, encoding=encoding
    )

    if lazy_decode:
      raise ValueError('lazy decoding not supported')

    super().__init__(
        file_format=None,
        shape=shape,
        dtype=tf.float32,
        sample_rate=sample_rate,
        encoding=encoding,
        doc=doc,
        lazy_decode=lazy_decode,
    )

  def get_serialized_info(self):
    return self._int16_tensor_feature.get_serialized_info()

  def encode_example(self, example_data):
    if not isinstance(example_data, np.ndarray):
      example_data = np.array(example_data, dtype=np.float32)
    if example_data.dtype != np.float32:
      raise ValueError('dtype should be float32')
    if example_data.min() < -1.0 or example_data.max() > 1.0 - (
        1.0 / self.INT16_SCALE
    ):
      raise ValueError('values should be in [-1, 1)')
    return self._int16_tensor_feature.encode_example(
        (example_data * self.INT16_SCALE).astype(np.int16)
    )

  def decode_example(self, tfexample_data):
    int16_scale = tf.constant(self.INT16_SCALE, dtype=tf.float32)
    decoded_data = tf.cast(
        self._int16_tensor_feature.decode_example(tfexample_data), tf.float32
    )
    return decoded_data / int16_scale
