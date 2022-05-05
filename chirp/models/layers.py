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

"""Model layers.

Building blocks and layers to construct networks, implemented as Flax modules.
"""
from typing import Callable, Optional, Tuple

from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp


class SqueezeAndExcitation(nn.Module):
  """Squeeze-and-Excitation layer.

  See "Squeeze-and-Excitation Networks" (Hu et al., 2018), particularly
  equations 2 and 3.

  Attributes:
    reduction_ratio: The reduction factor in the squeeze operation. Referred to
      as `r` in the paper.
    activation: The activation to apply after squeezing.
  """
  reduction_ratio: int = 4
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies SqueezeAndExcite on the 2D inputs.

    Args:
      inputs: Input data in shape of `(batch size, height, width, channels)`.

    Returns:
      JAX array with same shape as the input.
    """
    if inputs.ndim != 4:
      raise ValueError(
          "Inputs should in shape of `[batch size, height, width, features]`")

    # Squeeze
    x = jnp.mean(inputs, axis=(1, 2))
    x = nn.Dense(features=x.shape[-1] // self.reduction_ratio, name="Reduce")(x)
    x = self.activation(x)

    # Excite
    x = nn.Dense(features=inputs.shape[-1], name="Expand")(x)
    x = nn.sigmoid(x)
    return inputs * x[:, None, None, :]


class MBConv(nn.Module):
  """Mobile inverted bottleneck block.

  As introduced in "Mobilenetv2: Inverted residuals and linear bottlenecks"
  (Sandler et al., 2018). See figure 4d for an illustration and
  https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet for
  a reference implementation.

  The defaults are those from the MobileNetV2 paper. There is added support for
  batch normalization and squeeze-and-excitation blocks as used by EfficientNet.
  Note that the skip connection is not part of this module.

  Attributes:
    features: The number of filters.
    strides: The strides to use in the depthwise separable convolution.
    expand_ratio: The expansion factor to use. A block with expansion factor `N`
      is commonly referred to as MBConvN.
    kernel_size: The kernel size used by the depthwise separable convolution.
    activation: The activation function to use after the expanding 1x1
      convolution. Also used by the optional squeeze-and-excitation block.
    batch_norm: Whether to use batch normalization after the expanding and
      reducing convolutions.
    reduction_ratio: If given, a squeeze-and-excitation block is inserted after
      the depthwise separable convolution with the given reduction factor. Note
      that this reduction ratio is relative to the number of input channels,
      i.e., it scales with `expand_ratio`.
  """
  features: int
  strides: int
  expand_ratio: int
  kernel_size: Tuple[int, int] = (3, 3)
  activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu6
  batch_norm: bool = False
  reduction_ratio: Optional[int] = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Applies an inverted bottleneck block to the inputs.

    Args:
      inputs: Inputs should be of shape `(batch size, height, width, channels)`.
      train: Whether this is training (affects batch norm).

    Returns:
      A JAX array of `(batch size, height, width, features)`.
    """
    features = self.expand_ratio * inputs.shape[-1]

    x = inputs
    if self.expand_ratio != 1:
      x = nn.Conv(
          features=features,
          kernel_size=(1, 1),
          strides=(1, 1),
          use_bias=False,
          name="ExpandConv")(
              x)
      if self.batch_norm:
        x = nn.BatchNorm(
            use_running_average=not train, name="ExpandBatchNorm")(
                x)
      x = self.activation(x)

    if self.strides == 2:

      def _pad_width(input_size: int, kernel_size: int) -> Tuple[int, int]:
        """Calculate padding required to halve input with stride 2."""
        return (kernel_size // 2) - (1 - input_size % 2), kernel_size // 2

      padding = (_pad_width(x.shape[1], self.kernel_size[0]),
                 _pad_width(x.shape[2], self.kernel_size[1]))
    else:
      padding = "SAME"

    x = nn.Conv(
        features=features,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=padding,
        feature_group_count=features,
        use_bias=False,
        name="DepthwiseConv")(
            x)
    if self.batch_norm:
      x = nn.BatchNorm(
          use_running_average=not train, name="DepthwiseBatchNorm")(
              x)
    x = self.activation(x)

    if self.reduction_ratio is not None:
      x = SqueezeAndExcitation(
          reduction_ratio=self.reduction_ratio * self.expand_ratio,
          activation=self.activation)(
              x)
    x = nn.Conv(
        features=self.features,
        kernel_size=(1, 1),
        strides=1,
        use_bias=False,
        name="ProjectConv")(
            x)
    if self.batch_norm:
      x = nn.BatchNorm(
          use_running_average=not train, name="ProjectBatchNorm")(
              x)

    return x


class Identity(nn.Module):
  """Identity layer."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    """Identity function.

    Args:
      inputs: Input array.
      *args: Any other arguments are ignored.
      **kwargs: Any keyword arguments are ignored.

    Returns:
      The input, unchanged.
    """
    return inputs
