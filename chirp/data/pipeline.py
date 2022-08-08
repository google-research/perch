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

"""Data pipeline functions."""
import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Import bird_taxonomy to register the custom tfds.features.FeatureConnector.
import chirp.data.bird_taxonomy  # pylint: disable=unused-import
import jax
from jax import numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

_DEFAULT_DATASET_DIR = None
_DEFAULT_TFDS_DATADIR = None

Features = Dict[str, tf.Tensor]


class FeaturesPreprocessOp:

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    return features.copy()


class DatasetPreprocessOp:

  def __call__(self, dataset: tf.data.Dataset,
               dataset_info: tfds.core.DatasetInfo) -> tf.data.Dataset:
    return dataset


@dataclasses.dataclass
class Pipeline:
  """Construct a pipeline of preprocessing operations.

  This is modelled after `clu.preprocess_spec`, but rewritten to allow for
  processing operations which cannot be expressed per sample (e.g., mixing
  samples). Additionally, preprocessing operations will have access to the
  metadata in the DatasetInfo object.

  Attributes:
    ops: The preprocessing operations to apply.
    num_parallel_calls: Passed to `dataset.map`.
  """
  ops: Sequence[Union[FeaturesPreprocessOp, DatasetPreprocessOp]]
  num_parallel_calls: int = tf.data.AUTOTUNE

  def __call__(self, dataset: tf.data.Dataset,
               dataset_info: tfds.core.DatasetInfo) -> tf.data.Dataset:
    # We group feature preprocessing operations into a single map operation to
    # reduce the number of threads
    feature_preprocess_ops = []
    for op in self.ops:
      if isinstance(op, FeaturesPreprocessOp):
        feature_preprocess_ops.append(op)
      else:
        if feature_preprocess_ops:
          dataset = dataset.map(
              map_func=self.chain(feature_preprocess_ops, dataset_info),
              num_parallel_calls=self.num_parallel_calls)
          feature_preprocess_ops.clear()
        dataset = op(dataset, dataset_info)
    if feature_preprocess_ops:
      dataset = dataset.map(
          map_func=self.chain(feature_preprocess_ops, dataset_info),
          num_parallel_calls=self.num_parallel_calls)
    return dataset

  @staticmethod
  def chain(ops: Sequence[FeaturesPreprocessOp],
            dataset_info: tfds.core.DatasetInfo):

    def map_func(features: Features) -> Features:
      for op in ops:
        features = op(features, dataset_info)
      return features

    return map_func


@dataclasses.dataclass
class Slice(FeaturesPreprocessOp):
  """Slices a window of the input.

  Selects a window of the input data. Slices over the last axis.

  Attributes:
    window_size: The size of the window to take, in seconds.
    start: The starting point of the window, in seconds.
    names: The name of the features to slice. Each will be sliced the same way.
  """
  window_size: float
  start: float
  names: Tuple[str, ...] = ('audio', 'source_audio')

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    sample_rate = dataset_info.features[self.names[0]].sample_rate
    window_size = tf.cast(self.window_size * sample_rate, tf.int64)
    start = tf.cast(self.start * sample_rate, tf.int64)

    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = features[name][..., start:start + window_size]
    return features


@dataclasses.dataclass
class RandomSlice(FeaturesPreprocessOp):
  """Slices a random window of the input.

  Selects a random window of the input data. Slices over the last axis.

  Attributes:
    window_size: The size of the window to take, in seconds.
    names: The name of the features to slice. Each will be sliced the same way.
  """
  window_size: float
  names: Tuple[str, ...] = ('audio', 'source_audio')

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    sample_rate = dataset_info.features[self.names[0]].sample_rate
    audio_len = tf.shape(features[self.names[0]])[-1] / sample_rate
    max_start = tf.cast(audio_len - self.window_size, tf.float32)
    start = tf.random.uniform(shape=(), minval=0, maxval=max_start)

    return Slice(self.window_size, start, self.names)(features, dataset_info)


@dataclasses.dataclass
class NormalizeAudio(FeaturesPreprocessOp):
  """Normalize audio.

  Scales the signal so that the gain (maximum amplitude of the signal) is
  equal to the target gain. Assumes the signal is on the last axis.

  Attributes:
    target_gain: The target gain.
    names: The name of the features to normalize. The first will be used to
      calculate the normalization standard.
    eps: An epsilon that is used to avoid division by zero.
  """
  target_gain: float
  names: Tuple[str, ...] = ('audio', 'source_audio')
  eps: float = 0.01

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    del dataset_info  # Unused

    max_gain = tf.reduce_max(
        tf.abs(features[self.names[0]]), axis=-1, keepdims=True)
    gain_scalar = self.target_gain / (max_gain + self.eps)
    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = features[name] * tf.reshape(
          gain_scalar,
          tf.concat([
              tf.shape(gain_scalar),
              tf.ones([tf.rank(features[name]) - tf.rank(gain_scalar)],
                      dtype=tf.int32)
          ],
                    axis=0))
    return features


@dataclasses.dataclass
class RandomNormalizeAudio(FeaturesPreprocessOp):
  """Normalize audio using a random target gain.

  Scales the signal so that the gain (maximum amplitude of the signal) is
  equal to a target gain selected uniformly at random.

  Attributes:
    min_gain: The minimum target gain.
    max_gain: The minimum target gain.
    names: The name of the features to normalize. The first will be used to
      calculate the normalization standard.
    eps: An epsilon that is used to avoid division by zero.
  """
  min_gain: float
  max_gain: float
  names: Tuple[str, ...] = ('audio', 'source_audio')
  eps: float = 0.01

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    target_gain = tf.random.uniform([],
                                    minval=self.min_gain,
                                    maxval=self.max_gain)
    return NormalizeAudio(
        target_gain=target_gain, names=self.names, eps=self.eps)(features,
                                                                 dataset_info)


@dataclasses.dataclass
class MixAudio(DatasetPreprocessOp):
  """Mix audio samples.

  Attributes:
    mixin_prob: The probability with which samples are mixed. Note that if we
      mix, e.g., 50% of samples, the final ratio between mixed and unmixed
      samples is 1:2. More formally, to get a fraction `p` of the samples to be
      mixed, set `mixin_prob` to `2 * p / (p + 1)`.
    name: The name of the featuere to be mixed.
    source_name: The unmixed channels will be stored in this feature.
    pad_names: These labels must be padded to zeros.
    label_names: The names of the labels, which will be combined using an OR
      operation in the case of mixing.
    axis: The axis that should contain the mixed samples (for the source audio
      feature as well as the padded features). This should be set to the number
      of batch axes (e.g., 0 if this is applied before batching, 1 if applied
      after batching, and 2 if applied after batching with splitting across
      devices).
  """
  mixin_prob: float
  name: str = 'audio'
  source_name: str = 'source_audio'
  pad_names: Tuple[str, ...] = ('segment_start', 'segment_end')
  label_names: Tuple[str,
                     ...] = ('label', 'genus', 'family', 'order', 'bg_labels')
  axis: int = 0

  def __call__(self, dataset: tf.data.Dataset,
               dataset_info: tfds.core.DatasetInfo) -> tf.data.Dataset:
    del dataset_info  # Unused
    return dataset.group_by_window(
        self._key_func, self._reduce_func, window_size=2)

  def _key_func(self, features: Features) -> tf.Tensor:
    del features
    return tf.cast(tf.less(tf.random.uniform([]), self.mixin_prob), tf.int64)

  def _reduce_func(self, key: tf.Tensor,
                   dataset: tf.data.Dataset) -> tf.data.Dataset:
    key = tf.equal(key, 0)
    return tf.cond(
        key, lambda: dataset.batch(1, drop_remainder=True).map(self._mix_audio),
        lambda: dataset.batch(2, drop_remainder=True).map(self._mix_audio))

  @staticmethod
  def _pad_along_axis(tensor, paddings, axis, **kwargs):
    zero_paddings = tf.zeros([tf.rank(tensor), 2], dtype=tf.int32)
    paddings = tf.concat(
        [zero_paddings[:axis], [paddings], zero_paddings[axis + 1:]], axis=0)
    return tf.pad(tensor, paddings, **kwargs)

  def _mix_audio(self, features: Features) -> Features:
    """Mixes the samples."""
    for name in self.label_names:
      if name not in features:
        continue
      features[name] = tf.reduce_max(features[name], axis=0)

    source_audio = features[self.name]
    features[self.name] = tf.reduce_sum(source_audio, axis=0)

    # To enable batching we pad with zeros
    if source_audio.shape[0] == 1:
      source_audio = self._pad_along_axis(source_audio, [0, 1], axis=0)
      if self.axis:
        source_audio = tf.experimental.numpy.swapaxes(source_audio, 0,
                                                      self.axis)
      for name in self.pad_names:
        if name not in features:
          continue
        features[name] = self._pad_along_axis(features[name], [0, 1], axis=0)
        if self.axis:
          features[name] = tf.experimental.numpy.swapaxes(
              features[name], 0, self.axis)

    features[self.source_name] = source_audio
    return features


@dataclasses.dataclass
class MultiHot(FeaturesPreprocessOp):
  """Convert labels to multi-hot representation.

  This must be done before batching.

  Attributes:
    names: The labels to conver to multi-hot representations.
  """
  names: Tuple[str, ...] = ('label', 'genus', 'family', 'order', 'bg_labels')

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = tf.clip_by_value(
          tf.reduce_sum(
              tf.one_hot(
                  features[name],
                  dataset_info.features[name].feature.num_classes,
                  dtype=tf.int32),
              axis=0), 0, 1)

    return features


@dataclasses.dataclass
class OnlyJaxTypes(FeaturesPreprocessOp):
  """Discards tensors that are not supported by JAX (e.g., non-numeric).

  This must be done before batching.
  """

  def __call__(self, features: Features,
               dataset_info: tfds.core.DatasetInfo) -> Features:
    new_features = {}
    for name, feature in features.items():
      if isinstance(feature, tf.Tensor) and hasattr(
          jnp, feature.dtype.name) or feature.dtype is tf.bool:
        new_features[name] = feature
    return new_features


@dataclasses.dataclass
class Batch(DatasetPreprocessOp):
  """Collects samples into batches.

  This preprocessing operation drops the remainder by default.

  Attributes:
    batch_size: The batch size to use.
    split_across_devices: If true, the minibatch will be split into smaller
      minibatches to be distributed across the local devices present. This is
      useful for distributed training.
  """
  batch_size: int
  split_across_devices: bool = False

  def __call__(self, dataset: tf.data.Dataset,
               dataset_info: tfds.core.DatasetInfo) -> tf.data.Dataset:
    if self.split_across_devices:
      if self.batch_size % jax.local_device_count():
        raise ValueError('batch size must be divisible by number of devices')
      dataset = dataset.batch(
          self.batch_size // jax.local_device_count(), drop_remainder=True)
      return dataset.batch(jax.local_device_count(), drop_remainder=True)
    else:
      return dataset.batch(self.batch_size, drop_remainder=True)


def get_dataset(
    split: str,
    dataset_directory: str = _DEFAULT_DATASET_DIR,
    tfds_data_dir: Optional[str] = _DEFAULT_TFDS_DATADIR,
    tf_data_service_address: Optional[Any] = None,
    pipeline: Optional[Pipeline] = None,
    shuffle: Optional[bool] = None,
    repeat: Optional[bool] = None,
) -> Tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
  """Returns the placeholder dataset.

  Args:
    split: data split, e.g. 'train', 'test', 'train[:80%]', etc.
    dataset_directory: dataset directory.
    tfds_data_dir: If provided, uses tfds.add_data_dir, and then tfds.load,
      instead of using the tfds.builder_from_directory.
    tf_data_service_address: Address for TFDataService.
    pipeline: The preprocessing pipeline to apply to the data.
    shuffle: Whether to apply shuffling.
    repeat: Whether to repeat the dataset.

  Returns:
    The placeholder dataset.
  """
  if shuffle is None:
    shuffle = 'train' in split
  if repeat is None:
    repeat = 'train' in split

  if tfds_data_dir:
    tfds.core.add_data_dir(tfds_data_dir)
    ds, dataset_info = tfds.load(dataset_directory, split=split, with_info=True)
  else:
    builder = tfds.builder_from_directory(dataset_directory)
    ds = builder.as_dataset(split=split)
    dataset_info = builder.info
  if shuffle:
    ds = ds.shuffle(512)
  if pipeline is None:
    pipeline = Pipeline([
        OnlyJaxTypes(),
        MultiHot(),
        MixAudio(mixin_prob=0.25),
        Batch(8),
        RandomSlice(window_size=5),
        RandomNormalizeAudio(min_gain=0.15, max_gain=0.25),
    ])
  ds = pipeline(ds, dataset_info)

  if repeat:
    ds = ds.repeat()
  if 'train' in split and tf_data_service_address:
    ds = ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.DYNAMIC,
            service=tf_data_service_address,
            job_name='chirp_job'))
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds, dataset_info
