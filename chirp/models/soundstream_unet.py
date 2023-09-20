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

"""Adaptation of the Soundstream architecture for sound separation."""

from typing import Sequence

from flax import linen as nn
from jax import numpy as jnp

PADDING = "SAME"


class SeparableResnetBlock(nn.Module):
  """Resnet Block using a separable convolution.

  Attributes:
    num_hidden_filters: Number of hidden filters. If <0, uses the number of
      input filters.
    kernel_width: Width of depthwise convolutions.
    dilation: Convolution dilation.
    groups: Number of feature groups.
    residual_scalar: Scalar multiplier for residual connection.
    padding: Padding style.
  """

  num_hidden_filters: int
  kernel_width: int = 3
  dilation: int = 1
  groups: int = 1
  residual_scalar: float = 1.0
  padding: str = "SAME"

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    # First convolution is separable.
    input_dim = inputs.shape[-1]
    if self.num_hidden_filters < 0:
      num_hiddens = input_dim
    else:
      num_hiddens = self.num_hidden_filters

    x = nn.swish(inputs)
    x = nn.Conv(
        features=num_hiddens,
        kernel_size=(self.kernel_width,),
        strides=1,
        kernel_dilation=self.dilation,
        use_bias=True,
        feature_group_count=input_dim,
        padding=self.padding,
    )(x)
    x = nn.swish(x)
    x = nn.Conv(
        features=input_dim,
        kernel_size=(1,),
        strides=1,
        use_bias=True,
        feature_group_count=self.groups,
        padding=self.padding,
    )(x)
    return x + self.residual_scalar * inputs


class SeparatorBlock(nn.Module):
  """Block of residual layers and down/up-sampling layer.

  Attributes:
    is_encoder: Whether this is an encoder (downsampling) or decoder
      (upsampling) block.
    stride: Down/Up sample rate for this block.
    feature_mult: Multiplier/divisor for number of feature channels.
    groups: Number of feature groups.
    num_residual_layers: Number of dilated residual layers.
    num_residual_filters: Number of hidden residual filters.
    residual_kernel_width: Kernel width for residual layers.
    residual_scalar: Scaling constant for residual connections.
    padding: Padding style.
  """

  is_encoder: bool
  stride: int
  feature_mult: int
  groups: int = 1
  num_residual_layers: int = 3
  num_residual_filters: int = -1
  residual_kernel_width: int = 3
  residual_scalar: float = 1.0
  padding: str = "SAME"

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    input_dim = inputs.shape[-1]

    x = inputs
    if not self.is_encoder:
      # Decoder blocks start with up-sampling, for overall model symmetry.
      # Note that Linen's ConvTranspose doesn't support grouping, but it should
      # be easy to add support by exposing the feature_group_count argument
      # in lax's conv_general_dilated.
      x = nn.ConvTranspose(
          features=input_dim // self.feature_mult,
          kernel_size=(self.stride * 2,),
          strides=(self.stride,),
          use_bias=True,
          padding=self.padding,
      )(x)

    for idx in range(self.num_residual_layers):
      x = SeparableResnetBlock(
          num_hidden_filters=self.num_residual_filters,
          kernel_width=self.residual_kernel_width,
          dilation=self.residual_kernel_width**idx,
          groups=self.groups,
          residual_scalar=self.residual_scalar,
          padding=self.padding,
      )(x, train)
    x = nn.normalization.LayerNorm(reduction_axes=(-2, -1))(x)
    x = nn.swish(x)

    if self.is_encoder:
      x = nn.Conv(
          features=input_dim * self.feature_mult,
          kernel_size=(self.stride * 2,),
          strides=(self.stride,),
          use_bias=True,
          feature_group_count=self.groups,
          padding=self.padding,
      )(x)
    return x


class SoundstreamUNet(nn.Module):
  """Audio U-Net based on the Soundstream architecture.

  Assumes 1D inputs with shape [B, T, D].

  Attributes:
    base_filters: Number of filters for the input / output layer.
    bottleneck_filters: Number of filters in the inner bottleneck conv.
    output_filters: Number of filters in final model output.
    strides: Number of strides for each SeparatorBlock.
    feature_mults: Multiplier for number of features for each SeparatorBlock.
    groups: Number of feature groups for each SeparatorBlock.
    input_kernel_width: Width of the input convolution.
    bottleneck_kernel_width: Width of the bottleneck kernel.
    output_kernel_width: Width of the output kernel.
    num_residual_layers: Number of dilated residual layers per SeparatorBlock.
    residual_scalar: Scalar multiplier for residual connections.
    residual_hidden_filters: Number of hidden filters in residual blocks.
    unet_scalar: Scalar multiplier for UNet skip connections.
    padding: Padding style.
  """

  base_filters: int
  bottleneck_filters: int
  output_filters: int
  strides: Sequence[int]
  feature_mults: Sequence[int]
  groups: Sequence[int]
  input_kernel_width: int = 3
  bottleneck_kernel_width: int = 3
  output_kernel_width: int = 3
  num_residual_layers: int = 3
  residual_kernel_width: int = 3
  residual_scalar: float = 1.0
  residual_hidden_filters: int = -1
  # TODO(tomdenton): Experiment with a learnable scalar. See TDCN++.
  unet_scalar: float = 1.0
  padding: str = "SAME"

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Generate separation masks."""
    # Stem!
    x = nn.Conv(
        features=self.base_filters,
        kernel_size=(self.input_kernel_width,),
        strides=1,
        use_bias=True,
        padding=self.padding,
    )(inputs)
    x = nn.normalization.LayerNorm(reduction_axes=(-2, -1))(x)
    x = nn.swish(x)

    # Encoder!
    encoder_outputs = []
    for stride, mult, num_groups in zip(
        self.strides, self.feature_mults, self.groups
    ):
      x = SeparatorBlock(
          is_encoder=True,
          stride=stride,
          feature_mult=mult,
          groups=num_groups,
          num_residual_layers=self.num_residual_layers,
          num_residual_filters=self.residual_hidden_filters,
          residual_kernel_width=self.residual_kernel_width,
          residual_scalar=self.residual_scalar,
          padding=self.padding,
      )(x, train)
      encoder_outputs.append(x)

    # Bottleneck!
    prebottleneck_filters = x.shape[-1]
    x = nn.Conv(
        features=self.bottleneck_filters,
        kernel_size=(self.bottleneck_kernel_width,),
        strides=1,
        use_bias=True,
        padding=self.padding,
    )(x)

    # Normally this is where one would apply quantization for a codec.
    bottleneck_features = x

    # Unbottleneck!
    x = nn.Conv(
        features=prebottleneck_filters,
        kernel_size=(self.bottleneck_kernel_width,),
        strides=1,
        use_bias=True,
        padding=self.padding,
    )(x)

    # Decode!
    for stride, mult, num_groups, unet_features in zip(
        self.strides[::-1],
        self.feature_mults[::-1],
        self.groups[::-1],
        encoder_outputs[::-1],
    ):
      x = self.unet_scalar * unet_features + x
      x = SeparatorBlock(
          is_encoder=False,
          stride=stride,
          feature_mult=mult,
          groups=num_groups,
          num_residual_layers=self.num_residual_layers,
          num_residual_filters=self.residual_hidden_filters,
          residual_kernel_width=self.residual_kernel_width,
          residual_scalar=self.residual_scalar,
          padding=self.padding,
      )(x, train)

    # Head!
    x = nn.Conv(
        features=self.output_filters,
        kernel_size=(self.output_kernel_width,),
        strides=1,
        use_bias=True,
        padding=self.padding,
    )(x)

    return x, bottleneck_features  # pytype: disable=bad-return-type  # jax-ndarray
