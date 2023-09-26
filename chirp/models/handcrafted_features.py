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

"""Handcrafted features for linear models."""
from flax import linen as nn
from jax import numpy as jnp
from jax import scipy as jsp


class HandcraftedFeatures(nn.Module):
  """Handcrafted features for linear models.

  Attributes:
    compute_mfccs: If True, turn log-melspectrograms into MFCCs.
    num_mfccs: How many MFCCs to keep. Unused if compute_mfccs is False.
    aggregation: How to aggregate over time. If 'beans', we concatenate the
      mean, standard deviation, min, and max over the time axis (which mirrors
      the processing done in the BEANS benchmark (Hagiwara et al., 2022)). If
      'avg_pool', we perform average pooling over the time axis (controlled by
      `window_size` and `window_stride`) before flattening the time and channel
      axes. If `flatten`, we simply flatten the time and channel axes.
    window_size: Average pooling window size. Unused if `aggregation` is not
      `avg_pool`.
    window_stride: Average pooling window stride. Unused if `aggregation` is not
      `avg_pool`.
  """

  compute_mfccs: bool = False
  num_mfccs: int = 20
  aggregation: str = 'avg_pool'
  window_size: int = 10
  window_stride: int = 10

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
  ) -> jnp.ndarray:
    del train
    del use_running_average

    # Reshape from [B, T, D, 1] to [B, T, D]
    outputs = jnp.squeeze(inputs, axis=[-1])  # pytype: disable=wrong-arg-types  # jnp-type

    if self.compute_mfccs:
      outputs = jsp.fft.dct(
          outputs,
          type=2,
          n=self.num_mfccs,
          axis=-1,
          norm='ortho',
      )

    if self.aggregation == 'beans':
      return jnp.concatenate(
          [
              outputs.mean(axis=-2),
              outputs.std(axis=-2),
              outputs.min(axis=-2),
              outputs.max(axis=-2),
          ],
          axis=-1,
      )
    elif self.aggregation in ('flatten', 'avg_pool'):
      if self.aggregation == 'avg_pool':
        outputs = nn.pooling.avg_pool(
            outputs,
            window_shape=(self.window_size,),
            strides=(self.window_stride,),
        )
      # Reshape from [B, T, D] to [B, T * D]
      return outputs.reshape(
          outputs.shape[0], outputs.shape[1] * outputs.shape[2]
      )
    else:
      raise ValueError(f'unrecognized aggregation: {self.aggregation}')
