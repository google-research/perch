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

"""The ResNet v1.5 architecture used by NRC [1].

The architecture corresponds to the ResNet architecture in TorchVision, with the
following modifications:
  - A dense bottleneck layer followed by batch normalization is applied after
    the global average pooling operation.
  - The dense output layer is weight-normalized.

[1] Yang, Shiqi, et al. "Exploiting the intrinsic neighborhood structure for
source-free domain adaptation." Advances in Neural Information Processing
Systems 34 (2021): 29393-29405.
"""

import functools
import re

from chirp.models import output
from chirp.projects.sfda.models import resnet
from etils import epath
import flax
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class WNDense(nn.Dense):
  """Weight-normalized Dense layer."""

  def param(self, name, init_fn, *init_args):
    if name == 'kernel':
      kernel_v = super().param('kernel_v', init_fn, *init_args)
      param_shape, param_dtype = init_args
      param_shape = (1, param_shape[1])
      kernel_g = super().param('kernel_g', init_fn, *(param_shape, param_dtype))
      scale = jnp.sqrt(
          jnp.square(kernel_v).sum(
              tuple(range(kernel_v.ndim - 1)), keepdims=True))
      return kernel_g * kernel_v / scale
    else:
      return super().param(name, init_fn, *init_args)


class NRCResNet(resnet.ResNet):
  """Re-implementation of the ResNet v1.5 architecture used in NRC."""

  bottleneck_width: int = 256

  @nn.compact
  def __call__(self, x, train: bool, use_running_average: bool):

    # There *is* a computational difference between using padding='SAME' and
    # padding=1 for strided 3x3 convolutions, and to maintain compatibility
    # with the PyTorch implementation of ResNet we need to pass padding=1 rather
    # than the default padding='SAME' for 3x3 convolutions.
    def conv(*args, **kwargs):
      if args[1] == (3, 3):
        fn = functools.partial(
            self.conv, use_bias=False, padding=1, dtype=self.dtype)
      else:
        fn = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
      return fn(*args, **kwargs)

    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=use_running_average,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)

    x = conv(
        self.num_filters, (7, 7), (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init')(
            x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act)(
                x)
    x = jnp.mean(x, axis=(1, 2))

    x = nn.Dense(self.bottleneck_width, name='bottleneck_dense')(x)
    x = norm(name='bottleneck_bn')(x)

    model_outputs = {}
    model_outputs['embedding'] = x
    x = WNDense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    model_outputs['label'] = x.astype(jnp.float32)
    return output.ClassifierOutput(**model_outputs)

  @staticmethod
  def load_ckpt(dataset_name: str) -> flax.core.frozen_dict.FrozenDict:
    pretrained_ckpt_dir = NRCResNet.get_ckpt_path(dataset_name)
    with pretrained_ckpt_dir.open('rb') as f:
      state_dict = dict(np.load(f))
      variables = _to_variables(state_dict, dataset_name)
    return variables

  @staticmethod
  def get_ckpt_path(dataset_name: str) -> epath.Path:
    if 'vis_da_c' in dataset_name:
      # The public checkpoint doesn't exist because it's derived from a
      # PyTorch checkpoint (https://github.com/Albert0147/NRC_SFDA, which
      # points to the Google Drive directory https://drive.google.com/drive/
      # folders/1rI_I7GOHLi8jA4FnL10xdh8PA1bbwsIp).
      # Download the .pt files locally, then save them into a .npz file using
      # the following command:
      # state_dict = {}; load_fn = lambda letter: state_dict.update({
      #     f'{letter}.{k}': v for k, v in torch.load(
      #         f'source_{letter}.pt',
      #         map_location=torch.device('cpu')).items()})
      # load_fn('B'); load_fn('C'); load_fn('F')
      # np.savez('source.npz', **state_dict)
      # Finally, replace the '' below by the path to the source.npz file you
      # just created.
      return epath.Path('')
    else:
      raise NotImplementedError('No pretrained checkpoint available for '
                                f'dataset {dataset_name}.')

  @staticmethod
  def get_input_pipeline(data_builder: tfds.core.DatasetBuilder, split: str,
                         **kwargs) -> tf.data.Dataset:
    image_size = kwargs['image_size']
    padded_image_size = image_size + resnet.CROP_PADDING
    dtype = tf.float32
    read_config = tfds.ReadConfig(add_tfds_id=True)

    def process_example(example):
      image = example['image']

      # Resize and crop.
      image = example['image']
      image = tf.image.resize([image], [padded_image_size, padded_image_size],
                              method=tf.image.ResizeMethod.BILINEAR)[0]
      image = tf.image.central_crop(image, image_size / padded_image_size)

      # Reshape and normalize.
      image = tf.reshape(image, [image_size, image_size, 3])
      image -= tf.constant(resnet.MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(
          resnet.STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
      image = tf.image.convert_image_dtype(image, dtype=dtype)

      label = tf.one_hot(example['label'],
                         data_builder.info.features['label'].num_classes)

      return {'image': image, 'label': label, 'tfds_id': example['tfds_id']}

    dataset = data_builder.as_dataset(split=split, read_config=read_config)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)

    dataset = dataset.map(
        process_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def _to_variables(state_dict: dict[str, np.ndarray],
                  dataset_name: str) -> flax.core.scope.FrozenVariableDict:
  """Translates a PyTorch-style state dictionnary into a FrozenVariableDict.

  Args:
    state_dict: The PyTorch state_dict to translate.
    dataset_name: The name of the dataset the model was trained on, indicative
      of the model architecture used.

  Returns:
    The translated version of the state_dict.

  Raises:
    RuntimeError: If some convolutional kernel has neither 2 or 4 dimensions.
  """
  if dataset_name != 'vis_da_c':
    raise ValueError

  flat_params = {}
  flat_batch_stats = {}

  def _match_to_block_name(m):
    block_index_offsets = [0, 3, 7, 30]
    block_index = int(m.group(2)) + block_index_offsets[int(m.group(1)) - 1]
    return f'BottleneckResNetBlock_{block_index}.'

  renames = [
      # Groups and blocks
      functools.partial(
          re.compile(r'^F\.layer(\d*)\.(\d*)\.').sub,
          repl=_match_to_block_name),
      # Initial convolution
      functools.partial(re.compile(r'^F\.conv1').sub, repl=r'conv_init'),
      # Initial normalization
      functools.partial(re.compile(r'^F\.bn1').sub, repl=r'bn_init'),
      # Bottleneck
      functools.partial(
          re.compile(r'^B\.bottleneck').sub, repl=r'bottleneck_dense'),
      functools.partial(re.compile(r'^B\.bn').sub, repl=r'bottleneck_bn'),
      # Output layer
      functools.partial(re.compile(r'^C\.fc').sub, repl=r'WNDense_0'),
      # Convolutional layers
      functools.partial(
          re.compile(r'conv(\d)').sub,
          repl=lambda m: f'Conv_{int(m.group(1)) - 1}'),
      # Normalization layers
      functools.partial(
          re.compile(r'bn(\d)').sub,
          repl=lambda m: f'BatchNorm_{int(m.group(1)) - 1}'),
      # Downsampling layers
      functools.partial(
          re.compile(r'downsample\.(\d)').sub,
          repl=lambda m: 'norm_proj' if int(m.group(1)) else 'conv_proj'),
      # Normalization scaling coefficients. All other renamings of 'weight' map
      # to 'kernel', so we perform this renaming first.
      functools.partial(
          re.compile(r'BatchNorm_(\d)\.weight').sub,
          repl=r'BatchNorm_\1.scale'),
      functools.partial(
          re.compile(r'bn_init\.weight').sub, repl=r'bn_init.scale'),
      functools.partial(
          re.compile(r'norm_proj\.weight').sub, repl=r'norm_proj.scale'),
      functools.partial(
          re.compile(r'bottleneck_bn\.weight').sub,
          repl=r'bottleneck_bn.scale'),
      # Convolutional kernels
      functools.partial(re.compile(r'weight').sub, repl=r'kernel'),
      # Batch statistics
      functools.partial(re.compile(r'running_mean').sub, repl=r'mean'),
      functools.partial(re.compile(r'running_var').sub, repl=r'var'),
  ]

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


NRCResNet101 = functools.partial(
    NRCResNet,
    stage_sizes=[3, 4, 23, 3],
    block_cls=resnet.BottleneckResNetBlock)
