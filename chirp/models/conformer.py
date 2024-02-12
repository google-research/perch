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

"""Conformer layers."""
import dataclasses
import math
from typing import Callable

from chirp.models import layers
from flax import linen as nn
from jax import numpy as jnp
import numpy as np


class Conformer(nn.Module):
  """Projection layer followed by a conformer layer."""

  model_dims: int = 512
  kernel_size: int = 32
  ff_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  ff_residual_weight: float = 0.5
  ffn_dim_multiplier: int = 4
  atten_num_heads: int = 8
  layer_order: str = 'mhsa_before_conv'
  dropout_prob: float | None = None
  conv_residual_dropout: float | None = None
  atten_residual_dropout: float | None = None
  ffn_residual_dropout: float | None = None
  atten_dropout: float | None = None
  ffn_relu_dropout: float | None = None
  fflayer_weight_sharing: bool = False
  num_blocks: int = 1
  # tuples of layer index and corresponding scaling of number of channels
  downsample: list[tuple[int, float]] = dataclasses.field(default_factory=list)
  skip_layer_norm: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      return_intermediate_list: bool,
      use_running_average: bool | None = None,
  ) -> jnp.ndarray:
    """Projection followed by a conformer layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      train: Whether this is training. This affects Dropout behavior, and also
        affects BatchNorm behavior if 'use_running_average' is set to None.
      return_intermediate_list: Whether to return a list of the activations
        after each conformer block, instead of only the final ones.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.

    Returns:
      The conformer output with shape [B, T, D].
    """
    if use_running_average is None:
      use_running_average = not train
    if inputs.shape[-1] != self.model_dims:
      # Conformer requires the input dims to be `model_dims` so use a projection
      # layer that maps `input_dims` to `model_dims` before the conformer layer.
      inputs = layers.FeedForward(output_dims=self.model_dims)(inputs)

    if self.dropout_prob is not None:
      all_dropouts = [
          self.atten_dropout,
          self.atten_residual_dropout,
          self.conv_residual_dropout,
          self.ffn_residual_dropout,
          self.ffn_relu_dropout,
      ]
      for prob in all_dropouts:
        assert prob is None or prob == self.dropout_prob

      atten_dropout = self.dropout_prob
      atten_residual_dropout = self.dropout_prob
      conv_residual_dropout = self.dropout_prob
      ffn_residual_dropout = self.dropout_prob
      ffn_relu_dropout = self.dropout_prob
    else:
      atten_dropout = self.atten_dropout
      atten_residual_dropout = self.atten_residual_dropout
      conv_residual_dropout = self.conv_residual_dropout
      ffn_residual_dropout = self.ffn_residual_dropout
      ffn_relu_dropout = self.ffn_relu_dropout

    intermediate = []
    model_dims = self.model_dims
    downsample = list(self.downsample).copy()
    for i in range(self.num_blocks):
      if downsample and downsample[0][0] == i:
        should_downsample = True
        model_dims = int(model_dims * self.downsample[0][1])
        model_dims = (model_dims // self.atten_num_heads) * self.atten_num_heads
        downsample = downsample[1:]
      else:
        should_downsample = False
      inputs = layers.Conformer(
          model_dims=model_dims,
          kernel_size=self.kernel_size,
          ff_activation=self.ff_activation,
          ff_residual_weight=self.ff_residual_weight,
          ffn_dim_multiplier=self.ffn_dim_multiplier,
          atten_num_heads=self.atten_num_heads,
          layer_order=self.layer_order,
          dropout_prob=self.dropout_prob,
          conv_residual_dropout=conv_residual_dropout,
          atten_residual_dropout=atten_residual_dropout,
          ffn_residual_dropout=ffn_residual_dropout,
          atten_dropout=atten_dropout,
          ffn_relu_dropout=ffn_relu_dropout,
          fflayer_weight_sharing=self.fflayer_weight_sharing,
          name='conformer_block_{}'.format(i),
          downsample=should_downsample,
          skip_layer_norm=self.skip_layer_norm,
      )(inputs, train, use_running_average=use_running_average)
      intermediate.append(inputs)
    if return_intermediate_list:
      return intermediate  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      return inputs


class PositionalEmbedding(nn.Module):
  """Generates position embedding for a given 1-d sequence.

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """

  embedding_dims: int = 0
  min_timescale: int = 1
  max_timescale: int = 10_000

  @nn.compact
  def __call__(self, seq_length: int) -> jnp.ndarray:
    """Generates an array of sinusoids with different frequencies.

    Args:
      seq_length: Sequence length of the embeddings to be generated.

    Returns:
      An array of shape (1, seq_length, embedding_dim) containing positional
      embeddings.
    """
    position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    num_timescales = self.embedding_dims // 2
    log_timescale_increment = math.log(
        self.max_timescale / self.min_timescale
    ) / jnp.maximum(num_timescales - 1, 1.0)
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
    )
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    if self.embedding_dims != 0:
      signal = jnp.pad(
          signal, [(0, 0), (0, 0), (0, np.mod(self.embedding_dims, 2))]
      )
    return signal


class ConvolutionalSubsampling(nn.Module):
  """Convolutional subsampling module.

  This is the convolutional subsampling module as used in the conformer
  paper[^1]. It consists of two 2D convolutional layers with a stride of 2.
  The frequencies and output channels get combined to produce a 1D output.
  Relative positional embeddings are added for the conformer blocks.

  [1]: Gulati, Anmol, et al. "Conformer: Convolution-augmented transformer for
    speech recognition." arXiv preprint arXiv:2005.08100 (2020).
  """

  features: int
  kernel_size: tuple[int, int] = (3, 3)
  strides: tuple[int, int] = (2, 2)
  num_layers: int = 2
  dropout_prob: float = 0.1

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Apply convolutional subsampling.

    Args:
      inputs: A batch of spectrograms of size (batch, time, channels).
      train: Whether or not this is training (used for dropout).

    Returns:
      A subsampled array that is 4 times small in the time and channels dims.
    """
    x = inputs

    # Subsample
    x = nn.Conv(self.features, self.kernel_size, strides=self.strides)(x)
    x = nn.relu(x)
    x = nn.Conv(self.features, self.kernel_size, strides=self.strides)(x)
    x = nn.relu(x)

    # Merge channels and frequency dimension
    x = jnp.reshape(x, x.shape[:-2] + (-1,))
    x = nn.Dense(self.features)(x)

    # Add positional embeddings
    x = x + PositionalEmbedding(embedding_dims=self.features)(x.shape[-2])
    x = nn.Dropout(self.dropout_prob, deterministic=not train)(x)

    return x
