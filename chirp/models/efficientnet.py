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

"""EfficientNet.

Implementation of the EfficientNet model in Flax.
"""
import dataclasses
import enum
import math
from typing import Callable, NamedTuple

from aqt.jax.v2 import aqt_conv_general
from aqt.jax.v2 import config as aqt_cfg  # pylint: disable=unused-import
from chirp.models import layers
from flax import linen as nn
import flax.typing as flax_typing
import jax
from jax import numpy as jnp


class EfficientNetModel(enum.Enum):
  """Different variants of EfficientNet."""

  B0 = "b0"
  B1 = "b1"
  B2 = "b2"
  B3 = "b3"
  B4 = "b4"
  B5 = "b5"
  B6 = "b6"
  B7 = "b7"
  B8 = "b8"
  L2 = "l2"


class EfficientNetStage(NamedTuple):
  """Definition of a single stage in EfficientNet."""

  num_blocks: int
  features: int
  kernel_size: tuple[int, int]
  strides: int
  expand_ratio: int


# The values for EfficientNet-B0. The other variants are scalings of these.
# See table 1 in the paper or
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py
STEM_FEATURES = 32
STAGES = [
    EfficientNetStage(1, 16, (3, 3), 1, 1),
    EfficientNetStage(2, 24, (3, 3), 2, 6),
    EfficientNetStage(2, 40, (5, 5), 2, 6),
    EfficientNetStage(3, 80, (3, 3), 2, 6),
    EfficientNetStage(3, 112, (5, 5), 1, 6),
    EfficientNetStage(4, 192, (5, 5), 2, 6),
    EfficientNetStage(1, 320, (3, 3), 1, 6),
]
HEAD_FEATURES = 1280
REDUCTION_RATIO = 4


class EfficientNetScaling(NamedTuple):
  """Scaling for different model variants."""

  width_coefficient: float
  depth_coefficient: float
  dropout_rate: float


SCALINGS = {
    EfficientNetModel.B0: EfficientNetScaling(1.0, 1.0, 0.2),
    EfficientNetModel.B1: EfficientNetScaling(1.0, 1.1, 0.2),
    EfficientNetModel.B2: EfficientNetScaling(1.1, 1.2, 0.3),
    EfficientNetModel.B3: EfficientNetScaling(1.2, 1.4, 0.3),
    EfficientNetModel.B4: EfficientNetScaling(1.4, 1.8, 0.4),
    EfficientNetModel.B5: EfficientNetScaling(1.6, 2.2, 0.4),
    EfficientNetModel.B6: EfficientNetScaling(1.8, 2.6, 0.5),
    EfficientNetModel.B7: EfficientNetScaling(2.0, 3.1, 0.5),
    EfficientNetModel.B8: EfficientNetScaling(2.2, 3.6, 0.5),
    EfficientNetModel.L2: EfficientNetScaling(4.3, 5.3, 0.5),
}


def round_features(
    features: int, width_coefficient: float, depth_divisor: int = 8
) -> int:
  """Round number of filters based on width multiplier."""
  features *= width_coefficient
  new_features = max(
      depth_divisor,
      int(features + depth_divisor / 2) // depth_divisor * depth_divisor,
  )
  if new_features < 0.9 * features:
    new_features += depth_divisor
  return int(new_features)


def round_num_blocks(num_blocks: int, depth_coefficient: float) -> int:
  """Round number of blocks based on depth multiplier."""
  return int(math.ceil(depth_coefficient * num_blocks))


@dataclasses.dataclass
class OpSet:
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  sigmoid: Callable[[jnp.ndarray], jnp.ndarray] = nn.sigmoid
  stem_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  head_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  dot_general: flax_typing.DotGeneralT | None = None
  conv_general_dilated: flax_typing.ConvGeneralDilatedT | None = None


op_sets = {
    "default": OpSet(),
    "qat": OpSet(
        activation=nn.relu,
        sigmoid=nn.hard_sigmoid,
        stem_activation=nn.hard_swish,
        head_activation=nn.hard_swish,
        dot_general=jax.lax.dot_general,
        conv_general_dilated=aqt_conv_general.make_conv_general_dilated(
            aqt_cfg.conv_general_dilated_make(spatial_dimensions=2)
        ),
    ),
}


class Stem(nn.Module):
  """The stem of an EfficientNet model.

  The stem is the first layer, which is equivalent for all variations of
  EfficientNet.

  Attributes:
    features: The number of filters.
  """

  features: int
  conv_general_dilated: flax_typing.ConvGeneralDilatedT | None = None
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, use_running_average: bool
  ) -> jnp.ndarray:
    """Applies the first step of EfficientNet to the inputs.

    Args:
      inputs: Inputs should be of shape `(batch size, height, width, channels)`.
      use_running_average: Used to decide whether to use running statistics in
        BatchNorm (test mode), or the current batch's statistics (train mode).

    Returns:
      A JAX array of `(batch size, height, width, features)`.
    """
    x = nn.Conv(
        features=self.features,
        kernel_size=(3, 3),
        strides=2,
        use_bias=False,
        conv_general_dilated=self.conv_general_dilated,
        padding="VALID",
    )(inputs)
    x = nn.BatchNorm(use_running_average=use_running_average)(x)
    x = self.activation(x)
    return x


class Head(nn.Module):
  """The head of an EfficientNet model.

  The head is the last layer, which is equivalent for all variations of
  EfficientNet.

  Attributes:
    features: The number of filters.
    conv_general_dilated: Convolution op.
  """

  features: int
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
  conv_general_dilated: flax_typing.ConvGeneralDilatedT | None = None

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, use_running_average: bool
  ) -> jnp.ndarray:
    """Applies the last step of EfficientNet to the inputs.

    Args:
      inputs: Inputs should be of shape `(batch size, height, width, channels)`.
      use_running_average: Used to decide whether to use running statistics in
        BatchNorm (test mode), or the current batch's statistics (train mode).

    Returns:
      A JAX array of `(batch size, height, width, features)`.
    """
    x = nn.Conv(
        features=self.features,
        kernel_size=(1, 1),
        strides=1,
        use_bias=False,
        conv_general_dilated=self.conv_general_dilated,
    )(inputs)
    x = nn.BatchNorm(use_running_average=use_running_average)(x)
    x = self.activation(x)
    return x


class EfficientNet(nn.Module):
  """EfficientNet model.

  Attributes:
    model: The variant of EfficientNet model to use.
    include_top: If true, the model applies average pooling, flattens the
      output, and applies dropout. Note that this is different from Keras's
      `include_top` argument, which applies an additional linear transformation.
    survival_probability: The survival probability to use for stochastic depth.
    head: Optional Flax module to use as custom head.
    stem: Optional Flax module to use as custom stem.
    op_set: Named set of ops to use.
  """

  model: EfficientNetModel
  include_top: bool = True
  survival_probability: float = 0.8
  head: nn.Module | None = None
  stem: nn.Module | None = None
  op_set: str = "default"

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
  ) -> jnp.ndarray:
    """Applies EfficientNet to the inputs.

    Note that this model does not include the final pooling and fully connected
    layers.

    Args:
      inputs: Inputs should be of shape `(batch size, height, width, channels)`.
      train: Whether this is training. This affects Dropout behavior, and also
        affects BatchNorm behavior if 'use_running_average' is set to None.
      use_running_average: Optional, used to decide whether to use running
        statistics in BatchNorm (test mode), or the current batch's statistics
        (train mode). If not specified (or specified to None), default to 'not
        train'.

    Returns:
      A JAX array of `(batch size, height, width, features)` if `include_top` is
      false. If `include_top` is true the output is `(batch_size, features)`.
    """
    ops = op_sets[self.op_set]

    if use_running_average is None:
      use_running_average = not train
    scaling = SCALINGS[self.model]

    if self.stem is None:
      features = round_features(STEM_FEATURES, scaling.width_coefficient)
      stem = Stem(
          features,
          activation=ops.stem_activation,
          conv_general_dilated=ops.conv_general_dilated,
      )
    else:
      stem = self.stem

    x = stem(inputs, use_running_average=use_running_average)

    for stage in STAGES:
      num_blocks = round_num_blocks(stage.num_blocks, scaling.depth_coefficient)
      for block in range(num_blocks):
        # MBConv block with squeeze-and-excitation
        strides = stage.strides if block == 0 else 1
        features = round_features(stage.features, scaling.width_coefficient)
        mbconv = layers.MBConv(
            features=features,
            strides=strides,
            expand_ratio=stage.expand_ratio,
            kernel_size=stage.kernel_size,
            batch_norm=True,
            reduction_ratio=REDUCTION_RATIO,
            activation=ops.activation,
            sigmoid_activation=ops.sigmoid,
            dot_general=ops.dot_general,
            conv_general_dilated=ops.conv_general_dilated,
        )
        y = mbconv(x, train=train, use_running_average=use_running_average)

        # Stochastic depth
        if block > 0 and self.survival_probability:
          y = nn.Dropout(
              1 - self.survival_probability,
              broadcast_dims=(1, 2, 3),
              deterministic=not train,
          )(y)

        # Skip connections
        x = y if block == 0 else y + x

    if self.head is None:
      features = round_features(HEAD_FEATURES, scaling.width_coefficient)
      head = Head(
          features,
          activation=ops.head_activation,
          conv_general_dilated=ops.conv_general_dilated,
      )
    else:
      head = self.head

    x = head(x, use_running_average=use_running_average)

    if self.include_top:
      x = jnp.mean(x, axis=(1, 2))
      x = nn.Dropout(rate=scaling.dropout_rate, deterministic=not train)(x)

    return x
