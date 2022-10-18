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

"""The ResNet v1 architecture, borrowed from flax examples/ directory.

Taken from https://github.com/google/flax/blob/main/examples/imagenet/models.py.
 We make the following modifications to the orignal model:
  - Use of 'use_running_average' to explicitly control BatchNorm's behavior.
  - Packaging the forward's output as a taxonomy_model.ModelOutputs for
    compatibility with the rest of the pipeline.
  - Added a Dropout layer after average pooling, and before the classfication
    head. This was done to inject noise during the forward pass for Dropout
    Student and NOTELA.
  - Added a 'load_ckpt' method, following image_model.ImageModel template.
  - Added a 'get_ckpt_path' method, following image_model.ImageModel template.
  - Integrated flax's input pipeline in the 'get_input_pipeline' method.
"""

import functools
from typing import Any, Callable, Sequence, Tuple

from chirp.models import taxonomy_model
from chirp.projects.sfda.models import image_model
from etils import epath
import flax
from flax import linen as nn
from flax.training import checkpoints as flax_checkpoints
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

ModuleDef = Any
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable[[jnp.ndarray], jnp.ndarray]
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj')(
              residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable[[jnp.ndarray], jnp.ndarray]
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name='conv_proj')(
              residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(image_model.ImageModel):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: jnp.dtype = jnp.float32
  act: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool, use_running_average: bool):
    conv = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
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
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
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

    # The following Dropout was added to inject noise during foward pass.
    x = nn.Dropout(0.1, deterministic=not train)(x)

    model_outputs = {}
    model_outputs['embedding'] = x
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    model_outputs['label'] = x.astype(jnp.float32)
    return taxonomy_model.ModelOutputs(**model_outputs)

  @staticmethod
  def load_ckpt(dataset_name: str) -> flax.core.frozen_dict.FrozenDict:
    pretrained_ckpt_dir = ResNet.get_ckpt_path(dataset_name)
    state_dict = flax_checkpoints.restore_checkpoint(
        pretrained_ckpt_dir, target=None)
    variables = flax.core.freeze({
        'params': state_dict['params'],
        'batch_stats': state_dict['batch_stats']
    })
    return variables

  @staticmethod
  def get_ckpt_path(dataset_name: str) -> epath.Path:
    if 'imagenet' in dataset_name:
      return epath.Path(
          'gs://flax_public/examples/imagenet/v100_x8/checkpoint_250200')
    else:
      raise NotImplementedError('No pretrained checkpoint available for '
                                f'dataset {dataset_name}.')

  @staticmethod
  def get_input_pipeline(data_builder: tfds.core.DatasetBuilder, split: str,
                         **kwargs) -> tf.data.Dataset:
    image_size = kwargs['image_size']
    dtype = tf.float32
    read_config = tfds.ReadConfig(add_tfds_id=True)

    def _resize(image):
      return tf.image.resize([image], [image_size, image_size],
                             method=tf.image.ResizeMethod.BICUBIC)[0]

    def _decode_and_center_crop(image_bytes):
      """Crops to center of image with padding then scales image_size."""
      shape = tf.io.extract_jpeg_shape(image_bytes)
      image_height = shape[0]
      image_width = shape[1]

      padded_center_crop_size = tf.cast(
          ((image_size / (image_size + CROP_PADDING)) *
           tf.cast(tf.minimum(image_height, image_width), tf.float32)),
          tf.int32)

      offset_height = ((image_height - padded_center_crop_size) + 1) // 2
      offset_width = ((image_width - padded_center_crop_size) + 1) // 2
      crop_window = tf.stack([
          offset_height, offset_width, padded_center_crop_size,
          padded_center_crop_size
      ])
      image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
      image = _resize(image)

      return image

    def normalize_image(image):
      image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
      return image

    def preprocess_for_eval(image_bytes, dtype=tf.float32):
      """Preprocesses the given image for evaluation.

      Args:
        image_bytes: `Tensor` representing an image binary of arbitrary size.
        dtype: data type of the image.

      Returns:
        A preprocessed image `Tensor`.
      """
      image = _decode_and_center_crop(image_bytes)
      image = tf.reshape(image, [image_size, image_size, 3])
      image = normalize_image(image)
      image = tf.image.convert_image_dtype(image, dtype=dtype)
      return image

    def decode_example(example):
      image = preprocess_for_eval(example['image'], dtype=dtype)
      label = tf.one_hot(example['label'],
                         data_builder.info.features['label'].num_classes)
      return {'image': image, 'label': label}

    dataset = data_builder.as_dataset(
        split=split,
        decoders={'image': tfds.decode.SkipDecoding()},
        read_config=read_config)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


ResNet18 = functools.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = functools.partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)
ResNet101 = functools.partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock)
ResNet152 = functools.partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock)
ResNet200 = functools.partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock)

ResNet18Local = functools.partial(
    ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock, conv=nn.ConvLocal)
