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

"""EfficientNet v2 implementation, adapted from Keras."""

import dataclasses
import re

from chirp.models import efficientnet
from chirp.models import layers
from flax import linen as nn
from jax import numpy as jnp


#################### EfficientNet V2 configs ####################
v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]


v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]


v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]


v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]
efficientnetv2_params = {
    # (block, dropout)
    'efficientnetv2-s': (v2_s_block, 0.2),
    'efficientnetv2-m': (v2_m_block, 0.3),
    'efficientnetv2-l': (v2_l_block, 0.4),
    'efficientnetv2-xl': (v2_xl_block, 0.4),
}


@dataclasses.dataclass
class BlockArgs:
  num_repeats: int
  kernel_size: tuple[int, int]
  strides: int
  expand_ratio: int
  input_filters: int
  output_filters: int
  conv_type: int
  # reduction_ratio == se_ratio in Keras codebase.
  reduction_ratio: int | None


def decode_block_string(block_string: str) -> BlockArgs:
  """Gets a block through a string notation of arguments."""
  assert isinstance(block_string, str)
  ops = block_string.split('_')
  options = {}
  for op in ops:
    match = re.match(r'([a-z]+)([\d\.]+)', op)
    if match is None:
      raise ValueError('Invalid block string: %s' % block_string)
    key, value = match.groups()
    options[key] = value

  return BlockArgs(
      kernel_size=(int(options['k']), int(options['k'])),
      num_repeats=int(options['r']),
      input_filters=int(options['i']),
      output_filters=int(options['o']),
      expand_ratio=int(options['e']),
      reduction_ratio=float(options['se']) if 'se' in options else None,
      strides=int(options['s']),
      conv_type=int(options['c']) if 'c' in options else 0,
  )


def make_conv_block(
    block_args: BlockArgs, ops: efficientnet.OpSet, stride: int | None = None
) -> nn.Module:
  """Create an MBConvBlock from BlockArgs."""
  args = dataclasses.asdict(block_args)
  if block_args.conv_type == 0:
    cls = layers.MBConv
  elif block_args.conv_type == 1:
    cls = layers.FusedMBConv
  else:
    raise ValueError('Unsupported conv_type: %d' % block_args.conv_type)
  args.pop('conv_type')

  # Convert se from float to reduction_ratio int.
  if block_args.reduction_ratio is not None:
    args['reduction_ratio'] = int(1.0 / block_args.reduction_ratio)

  if stride is not None:
    args['strides'] = stride
  args['features'] = args.pop('output_filters')
  # Input filters don't need to be pre-specified.
  args.pop('input_filters')
  args.pop('num_repeats')
  args['batch_norm'] = True
  args['activation'] = ops.activation
  args['sigmoid_activation'] = ops.sigmoid
  args['dot_general'] = ops.dot_general
  args['conv_general_dilated'] = ops.conv_general_dilated
  return cls(**args)


class EfficientNetV2(nn.Module):
  """EfficientNetV2 model with quantization-aware training.

  Attributes:
    model_name: The variant of EfficientNetV2 model to use.
    include_top: If true, the model applies average pooling, flattens the
      output, and applies dropout. Note that this is different from Keras's
      `include_top` argument, which applies an additional linear transformation.
    head: Optional Flax module to use as custom head.
    stem: Optional Flax module to use as custom stem.
    survival_probability: The survival probability to use for stochastic depth.
    dropout_rate: Dropout rate for convolutional features.
    op_set: Named set of ops to use.
  """

  model_name: str = 'efficientnetv2-s'
  include_top: bool = True
  survival_probability: float = 0.8
  head: nn.Module | None = None
  stem: nn.Module | None = None
  dropout_rate: float | None = None
  block_configs: list[BlockArgs] | None = None
  op_set: str = 'default'

  def __post_init__(self):
    super().__post_init__()
    if self.block_configs is None:
      block_strs, dropout_rate = efficientnetv2_params[self.model_name]
      object.__setattr__(self, 'dropout_rate', dropout_rate)
      object.__setattr__(
          self, 'block_configs', [decode_block_string(s) for s in block_strs]
      )

  def residual(
      self,
      x: jnp.ndarray,
      y: jnp.ndarray,
      block_idx: int,
      block_strides: tuple[int, int],
      survival_prob: float,
      train: bool,
  ) -> jnp.ndarray:
    if not (block_strides == (1, 1) and x.shape[-1] == y.shape[-1]):
      return y
    if block_idx == 0:
      return y

    # Stochastic depth
    if survival_prob:
      y = nn.Dropout(
          1 - survival_prob,
          broadcast_dims=(1, 2, 3),
          deterministic=not train,
      )(y)
    # Skip connection
    return y + x

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
      use_running_average: bool | None = None,
  ) -> jnp.ndarray:
    ops = efficientnet.op_sets[self.op_set]

    if use_running_average is None:
      use_running_average = not train

    if self.stem is None:
      stem = efficientnet.Stem(
          features=self.block_configs[0].input_filters,
          activation=ops.stem_activation,
          conv_general_dilated=ops.conv_general_dilated,
      )
    else:
      stem = self.stem
    x = stem(inputs, use_running_average=use_running_average)

    for block_idx, block_config in enumerate(self.block_configs):

      if self.survival_probability:
        drop_rate = 1.0 - self.survival_probability
        survival_prob = 1.0 - drop_rate * float(block_idx) / len(
            self.block_configs
        )
      else:
        survival_prob = 0.0

      for i in range(block_config.num_repeats):
        # Stride is only applied in the first repeat.
        block_strides = (1, 1) if i > 0 else block_config.strides
        block = make_conv_block(block_config, ops, stride=block_strides)
        y = block(x, train=train, use_running_average=use_running_average)
        x = self.residual(x, y, block_idx, block_strides, survival_prob, train)

    head = self.head or efficientnet.Head(
        1280,
        activation=ops.head_activation,
        conv_general_dilated=ops.conv_general_dilated,
    )
    x = head(x, use_running_average=use_running_average)

    if self.include_top:
      x = jnp.mean(x, axis=(1, 2))
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    return x
