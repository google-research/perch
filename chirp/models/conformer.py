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

"""Conformer layers."""
from typing import Callable, Optional
from chirp.models import layers

from flax import linen as nn
from jax import numpy as jnp

JTensor = jnp.ndarray


class Conformer(nn.Module):
  """Projection layer followed by a conformer layer."""
  model_dims: int = 512
  kernel_size: int = 32
  ff_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  ff_residual_weight: float = 0.5
  ffn_dim_multiplier: int = 4
  atten_num_heads: int = 8
  layer_order: str = 'mhsa_before_conv'
  dropout_prob: Optional[float] = None
  conv_residual_dropout: Optional[float] = None
  atten_residual_dropout: Optional[float] = None
  ffn_residual_dropout: Optional[float] = None
  atten_dropout: Optional[float] = None
  ffn_relu_dropout: Optional[float] = None
  fflayer_weight_sharing: bool = False
  num_blocks: int = 1

  @nn.compact
  def __call__(self, inputs: JTensor, train: bool) -> JTensor:
    """Projection followed by a conformer layer.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      train: Whether we are in train mode. Affects dropout and batch norm.

    Returns:
      The conformer output with shape [B, T, D].
    """
    if inputs.shape[-1] != self.model_dims:
      # Conformer requires the input dims to be `model_dims` so use a projection
      # layer that maps `input_dims` to `model_dims` before the conformer layer.
      inputs = layers.FeedForward(output_dims=self.model_dims)(inputs)

    if self.dropout_prob is not None:
      all_dropouts = [
          self.atten_dropout, self.atten_residual_dropout,
          self.conv_residual_dropout, self.ffn_residual_dropout,
          self.ffn_relu_dropout
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

    for i in range(self.num_blocks):
      inputs = layers.Conformer(
          model_dims=self.model_dims,
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
          name='conformer_block_{}'.format(i))(inputs, train)
    return inputs
