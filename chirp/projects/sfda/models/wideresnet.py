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

"""The WideResnet architecture.

Translated from PyTorch version at https://github.com/RobustBench/robustbench/
blob/master/robustbench/model_zoo/architectures/wide_resnet.py. We make the
following modifications to the orignal model:
  - Use of 'use_running_average' to explicitly control BatchNorm's behavior.
  - Packaging the forward's output as a taxonomy_model.ModelOutputs for
    compatibility with the rest of the pipeline.
  - Added a Dropout layer after average pooling, and before the classfication
    head. This was done to inject noise during the forward pass for Dropout
    Student and NOTELA.
  - Added a 'load_ckpt' method, following image_model.ImageModel template.
  - Added a 'get_ckpt_path' method, following image_model.ImageModel template.
"""
import functools
import re
from typing import Dict, Tuple, List
from chirp.models import taxonomy_model
from chirp.projects.sfda.models import image_model
from etils import epath
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock.

  Attributes:
    channels: How many channels to use in the convolutional layers.
    strides: Strides for the pooling.
  """
  channels: int
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool,
               use_running_average: bool) -> jnp.ndarray:
    bn1 = nn.BatchNorm(use_running_average=use_running_average, name='norm_1')
    bn2 = nn.BatchNorm(use_running_average=use_running_average, name='norm_2')
    conv1 = nn.Conv(
        self.channels,
        kernel_size=(3, 3),
        strides=self.strides,
        padding=1,
        use_bias=False,
        name='conv1')
    conv2 = nn.Conv(
        self.channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=1,
        use_bias=False,
        name='conv2')

    if x.shape[-1] == self.channels:
      out = jax.nn.relu(bn1(x))
      out = jax.nn.relu(bn2(conv1(out)))
      out = conv2(out)
      out += x
    else:
      x = jax.nn.relu(bn1(x))
      out = jax.nn.relu(bn2(conv1(x)))
      out = conv2(out)
      out += nn.Conv(
          self.channels, (1, 1),
          self.strides,
          padding='VALID',
          use_bias=False,
          name='conv_shortcut')(
              x)

    return out


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup.

  Attributes:
    blocks_per_group: How many resnet blocks to add to each group (should be 4
      blocks for a WRN28, and 6 for a WRN40).
    channels: How many channels to use in the convolutional layers.
    strides: Strides for the pooling.
  """
  blocks_per_group: int
  channels: int
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool,
               use_running_average: bool) -> jnp.ndarray:
    for i in range(self.blocks_per_group):
      x = WideResnetBlock(self.channels, self.strides if i == 0 else
                          (1, 1))(x, train, use_running_average)
    return x


class WideResnet(image_model.ImageModel):
  """Defines the WideResnet Model.

  Attributes:
    blocks_per_group: How many resnet blocks to add to each group (should be 4
      blocks for a WRN28, and 6 for a WRN40).
    channel_multiplier: The multiplier to apply to the number of filters in the
      model (1 is classical resnet, 10 for WRN28-10, etc...).
    num_classes: Dimension of the output of the model (ie number of classes for
      a classification problem).
  """
  blocks_per_group: int
  channel_multiplier: int
  num_classes: int

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool,
               use_running_average: bool) -> jnp.ndarray:
    x = nn.Conv(16, (3, 3), padding=1, name='init_conv', use_bias=False)(x)
    x = WideResnetGroup(self.blocks_per_group, 16 * self.channel_multiplier)(
        x, train=train, use_running_average=use_running_average)
    x = WideResnetGroup(
        self.blocks_per_group,
        32 * self.channel_multiplier,
        (2, 2),
    )(x, train=train, use_running_average=use_running_average)
    x = WideResnetGroup(
        self.blocks_per_group,
        64 * self.channel_multiplier,
        (2, 2),
    )(x, train=train, use_running_average=use_running_average)
    x = jax.nn.relu(
        nn.BatchNorm(
            use_running_average=use_running_average, name='pre-pool-norm')(x))
    # The following Dropout was added to inject noise during foward pass.
    x = nn.Dropout(0.1, deterministic=not train)(x)
    x = nn.avg_pool(x, x.shape[1:3])
    x = x.reshape((x.shape[0], -1))

    model_outputs = {}
    model_outputs['embedding'] = x
    # Head
    outputs = nn.Dense(self.num_classes)(x)

    model_outputs['label'] = outputs.astype(jnp.float32)
    return taxonomy_model.ModelOutputs(**model_outputs)

  @staticmethod
  def load_ckpt(dataset_name: str) -> flax.core.frozen_dict.FrozenDict:
    pretrained_ckpt_dir = WideResnet.get_ckpt_path(dataset_name)
    with pretrained_ckpt_dir.open('rb') as f:
      state_dict = dict(np.load(f))
      variables = _to_variables(state_dict)
    return variables

  @staticmethod
  def get_ckpt_path(dataset_name: str) -> epath.Path:
    if 'cifar' in dataset_name:
      # The public checkpoint doesn't exist because it's derived from a
      # PyTorch checkpoint (https://github.com/RobustBench/robustbench/
      # blob/master/robustbench/model_zoo/cifar10.py#L760, which points to
      # https://drive.google.com/open?id= 1t98aEuzeTL8P7Kpd5DIrCoCL21BNZUhC).
      # Please, download this file, open it locally and save it into an
      # npz file using the following command:
      # np.savez(output_path, **torch.load(torch_checkpoint_path,
      # map_location=torch.device('cpu'))['state_dict'])
      # Finally, replace the '' below by the `output_path` above.
      return ''
    else:
      raise NotImplementedError('No pretrained checkpoint available for '
                                f'dataset {dataset_name}.')

  @staticmethod
  def is_bn_parameter(parameter_name: List[str]) -> bool:
    """Verifies whether some parameter belong to a BatchNorm layer.

    Only WideResnetGroup's BatchNorm parameters will be captured; the
    pre-pool-norm won't be included.

    Args:
      parameter_name: The name of the parameter, as a list in which each member
        describes the name of a layer. E.g. ('Block1', 'batch_norm_1', 'bias').

    Returns:
      True if this parameter belongs to a BatchNorm layer.
    """
    return any(['norm_' in x for x in parameter_name])


def _to_variables(
    state_dict: Dict[str, np.ndarray]) -> flax.core.scope.FrozenVariableDict:
  """Translates a PyTorch-style state dictionnary into a flax FrozenVariableDict.

  Args:
    state_dict: The PyTorch state_dict to translate.

  Returns:
    The translated version of the state_dict.

  Raises:
    RuntimeError: If some convolutional kernel has neither 2 or 4 dimensions.
  """
  flat_params = {}
  flat_batch_stats = {}

  renames = [
      # Groups
      functools.partial(
          re.compile(r'block(\d)').sub,
          repl=lambda m: f'WideResnetGroup_{int(m.group(1)) - 1}'),
      # Blocks
      functools.partial(
          re.compile(r'layer\.(\d)').sub, repl=r'WideResnetBlock_\1'),
      # Initial convolution
      functools.partial(re.compile(r'^conv1').sub, repl=r'init_conv'),
      # Pre-pooling normalization
      functools.partial(re.compile(r'^bn1').sub, repl=r'pre-pool-norm'),
      # Output layer
      functools.partial(re.compile(r'fc').sub, repl=r'Dense_0'),
      # Normalization layers
      functools.partial(re.compile(r'bn(\d)').sub, repl=r'norm_\1'),
      # Convolutional shortcut layers
      functools.partial(re.compile(r'convShortcut').sub, repl=r'conv_shortcut'),
      # Normalization scaling coefficients. All other renamings of 'weight' map
      # to 'kernel', so we perform this renaming first.
      functools.partial(
          re.compile(r'norm_(\d)\.weight').sub, repl=r'norm_\1.scale'),
      functools.partial(
          re.compile(r'pre-pool-norm.weight').sub, repl=r'pre-pool-norm.scale'),
      # Convolutional kernels
      functools.partial(re.compile(r'weight').sub, repl=r'kernel'),
      # Batch statistics
      functools.partial(re.compile(r'running_mean').sub, repl=r'mean'),
      functools.partial(re.compile(r'running_var').sub, repl=r'var'),
  ]
  transposes = [re.compile('kernel').search]  # pylint: disable=unused-variable

  for key, value in state_dict.items():
    # We don't need the 'num_batches_tracked' variables.
    if 'num_batches_tracked' in key:
      continue

    # Perform renaming.
    for rename in renames:
      key = rename(string=key)

    # Transpose convolutional kernels and weight matrices.
    if 'kernel' in key:
      if len(value.shape) == 2:
        value = value.transpose()
      elif len(value.shape) == 4:
        value = value.transpose(2, 3, 1, 0)
      else:
        raise RuntimeError

    # Route parameters and batch statistics to their appropriate flat
    # dictionary. Flax can unflatten dictionaries whose keys are tuples of
    # strings, which we take advantage of by splitting the keys by the '.'
    # character.
    flat_dict = (
        flat_batch_stats if 'mean' in key or 'var' in key else flat_params)
    flat_dict[tuple(key.split('.'))] = value

  return flax.core.freeze({
      'params': flax.traverse_util.unflatten_dict(flat_params),
      'batch_stats': flax.traverse_util.unflatten_dict(flat_batch_stats),
  })


WideResNet2810 = functools.partial(
    WideResnet, blocks_per_group=4, channel_multiplier=10)
