# coding=utf-8
# Copyright 2023 The Chirp Authors.
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
from typing import Iterable
from typing import Sequence
from absl import logging
from chirp import audio_utils
from chirp.models import frontend
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
import jax
from jax import numpy as jnp
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


Features = dict[str, tf.Tensor]


class FeaturesPreprocessOp:

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    return features.copy()


class DatasetPreprocessOp:

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
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
    deterministic: Whether the ordering of the samples should be deterministic.
  """

  ops: Sequence[FeaturesPreprocessOp | DatasetPreprocessOp]
  num_parallel_calls: int = tf.data.AUTOTUNE
  deterministic: bool = False

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
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
              num_parallel_calls=self.num_parallel_calls,
              deterministic=self.deterministic,
          )
          feature_preprocess_ops.clear()
        dataset = op(dataset, dataset_info)
    if feature_preprocess_ops:
      dataset = dataset.map(
          map_func=self.chain(feature_preprocess_ops, dataset_info),
          num_parallel_calls=self.num_parallel_calls,
          deterministic=self.deterministic,
      )
    return dataset

  @staticmethod
  def chain(
      ops: Sequence[FeaturesPreprocessOp], dataset_info: tfds.core.DatasetInfo
  ):
    def map_func(features: Features) -> Features:
      for op in ops:
        features = op(features, dataset_info)
      return features

    return map_func


@dataclasses.dataclass
class Pad(FeaturesPreprocessOp):
  """Pads the last axis to a minimum length.

  Attributes:
    pad_size: The minimum length to pad to.
    random: If true, pads a random amount left and right. If false, will pad the
      end only.
    add_mask: Whether to add a new mask feature indicating where the padding
      appears in the named features.
    names: The name of the features to pad.
  """

  pad_size: float
  random: bool = True
  add_mask: bool = True
  names: tuple[str, ...] = ('audio',)

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    sample_rate = dataset_info.features[self.names[0]].sample_rate
    window_size = tf.cast(self.pad_size * sample_rate, tf.int32)

    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      padding = tf.reduce_max([window_size - tf.shape(features[name])[-1], 0])
      if self.random:
        left_pad = tf.random.uniform(
            shape=(), minval=0, maxval=padding + 1, dtype=tf.int32
        )
        right_pad = padding - left_pad
      else:
        left_pad = 0
        right_pad = padding
      paddings = ((0, 0),) * (tf.rank(features[name]) - 1) + (
          (left_pad, right_pad),
      )

      mask = tf.ones_like(features[name])
      padded_mask = tf.pad(mask, paddings)
      if self.add_mask:
        features[f'{name}_mask'] = padded_mask

      features[name] = tf.pad(features[name], paddings)
    return features


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
  names: tuple[str, ...] = ('audio', 'source_audio', 'audio_mask')

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    sample_rate = dataset_info.features[self.names[0]].sample_rate
    window_size = tf.cast(self.window_size * sample_rate, tf.int64)
    start = tf.cast(self.start * sample_rate, tf.int64)

    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = features[name][..., start : start + window_size]
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
  names: tuple[str, ...] = ('audio', 'source_audio', 'audio_mask')

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
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
  names: tuple[str, ...] = ('audio', 'source_audio')
  eps: float = 0.01

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    del dataset_info  # Unused

    max_gain = tf.reduce_max(
        tf.abs(features[self.names[0]]), axis=-1, keepdims=True
    )
    gain_scalar = self.target_gain / (max_gain + self.eps)
    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = features[name] * tf.reshape(
          gain_scalar,
          tf.concat(
              [
                  tf.shape(gain_scalar),
                  tf.ones(
                      [tf.rank(features[name]) - tf.rank(gain_scalar)],
                      dtype=tf.int32,
                  ),
              ],
              axis=0,
          ),
      )
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
  names: tuple[str, ...] = ('audio', 'source_audio')
  eps: float = 0.01

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    target_gain = tf.random.uniform(
        [], minval=self.min_gain, maxval=self.max_gain
    )
    return NormalizeAudio(
        target_gain=target_gain, names=self.names, eps=self.eps
    )(features, dataset_info)


@dataclasses.dataclass
class MixAudio(DatasetPreprocessOp):
  """Mix audio samples.

  Attributes:
    mixin_prob: The probability of mixing a single example with a single other
      example. For a probability p this results in an unnormalized target
      distribution of (1 - p, p / 2). If this is given, target_dist cannot be
      given and vice versa.
    target_dist: The target distribution of mixtures containing 1, 2, ...
      sources. Does not have to be normalized. For example, (1., 1.) will result
      in half of the examples being raw examples, and the other half being
      mixtures of two examples.
    name: The name of the featuere to be mixed.
    source_name: The unmixed channels will be stored in this feature.
    pad_names: These labels must be padded to zeros.
    label_names: The names of the labels and masks, which will be combined using
      an OR operation in the case of mixing.
    axis: The axis that should contain the mixed samples (for the source audio
      feature as well as the padded features). This should be set to the number
      of batch axes (e.g., 0 if this is applied before batching, 1 if applied
      after batching, and 2 if applied after batching with splitting across
      devices).
  """

  mixin_prob: float | None = None
  target_dist: tuple[float, ...] | None = None
  name: str = 'audio'
  source_name: str = 'source_audio'
  pad_names: tuple[str, ...] = (
      'segment_start',
      'segment_end',
      'recording_id',
      'segment_id',
  )
  label_names: tuple[str, ...] = (
      'label',
      'genus',
      'family',
      'order',
      'bg_labels',
      'label_mask',
      'genus_mask',
      'family_mask',
      'order_mask',
      'bg_labels_mask',
      'audio_mask',
  )
  axis: int = 0

  def __post_init__(self):
    if not (self.mixin_prob is None) ^ (self.target_dist is None):
      raise ValueError('either mixin_prob or target_dist must be set')
    if self.target_dist is None:
      self.target_dist = (1 - self.mixin_prob, self.mixin_prob / 2)

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    del dataset_info  # Unused
    return dataset.group_by_window(
        self._key_func, self._reduce_func, window_size_func=lambda i: i + 1
    )

  def _key_func(self, features: Features) -> tf.Tensor:
    del features
    target_dist = tf.constant(self.target_dist, dtype=tf.float32)
    sample_dist = target_dist * (
        tf.range(len(self.target_dist), dtype=tf.float32) + 1.0
    )
    return tf.squeeze(tf.random.categorical(tf.math.log([sample_dist]), 1))

  def _reduce_func(
      self, key: tf.Tensor, dataset: tf.data.Dataset
  ) -> tf.data.Dataset:
    key = tf.cast(key, tf.int32)
    # pylint: disable=g-long-lambda
    return tf.switch_case(
        key,
        [
            lambda i=i: dataset.batch(i + 1, drop_remainder=True).map(
                self._mix_audio
            )
            for i in range(len(self.target_dist))
        ],
    )

  @staticmethod
  def _pad_along_axis(tensor, paddings, axis, **kwargs):
    zero_paddings = tf.zeros([tf.rank(tensor), 2], dtype=tf.int32)
    paddings = tf.concat(
        [zero_paddings[:axis], [paddings], zero_paddings[axis + 1 :]], axis=0
    )
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
    if source_audio.shape[0] < len(self.target_dist):
      p = len(self.target_dist) - source_audio.shape[0]
      source_audio = self._pad_along_axis(source_audio, [0, p], axis=0)
      if self.axis:
        source_audio = tf.experimental.numpy.swapaxes(
            source_audio, 0, self.axis
        )
      for name in self.pad_names:
        if name not in features:
          continue
        features[name] = self._pad_along_axis(features[name], [0, p], axis=0)
        if self.axis:
          features[name] = tf.experimental.numpy.swapaxes(
              features[name], 0, self.axis
          )

    features[self.source_name] = source_audio
    return features


@dataclasses.dataclass
class MultiHot(FeaturesPreprocessOp):
  """Convert labels to multi-hot representation.

  This must be done before batching.

  Attributes:
    names: The labels to convert to multi-hot representations.
  """

  names: tuple[str, ...] = ('label', 'genus', 'family', 'order', 'bg_labels')

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = tf.clip_by_value(
          tf.reduce_sum(
              tf.one_hot(
                  features[name],
                  dataset_info.features[name].feature.num_classes,
                  dtype=tf.int32,
              ),
              axis=0,
          ),
          0,
          1,
      )

    return features


@dataclasses.dataclass
class MergeBackgroundLabels(FeaturesPreprocessOp):
  """Include background labels in the set of labels for each example."""

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features = features.copy()
    features['label'] = tf.clip_by_value(
        features['label'] + features['bg_labels'], 0, 1
    )
    features['label_mask'] = tf.clip_by_value(
        features['label_mask'] + features['bg_labels_mask'], 0, 1
    )
    return features


@dataclasses.dataclass
class AddChannel(FeaturesPreprocessOp):
  name: str = 'audio'

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features = features.copy()
    features[self.name] = tf.expand_dims(features[self.name], axis=-1)
    return features


@dataclasses.dataclass
class MelSpectrogram(FeaturesPreprocessOp):
  """Convert audio to a spectrogram.

  Attributes:
    features: The number of channels to create.
    kernel_size: The kernel size to use.
    stride: The stride to use.
    sample_rate: The sample rate of the original audio.
    freq_range: The frequency range to capture.
    name: The name of the feature to process.
    power: The power of the magnitude spectrogram.
    scaling_config: The magnitude scaling to use.
    nfft: Length of the FFT used, if a zero padded FFT is desired.
  """

  features: int
  kernel_size: int
  stride: int
  sample_rate: int
  freq_range: tuple[int, int]
  name: str = 'audio'
  power: float = 2.0
  scaling_config: frontend.ScalingConfig | None = None
  nfft: int | None = None

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features = features.copy()
    stfts = audio_utils.stft_tf(
        features[self.name],
        nperseg=self.kernel_size,
        noverlap=self.kernel_size - self.stride,
        nfft=self.nfft,
        padded=False,
    )
    if tf.shape(features[self.name])[-1] % self.stride == 0:
      stfts = stfts[..., :-1]
    stfts = tf.experimental.numpy.swapaxes(stfts, -1, -2)
    magnitude_spectrograms = tf.math.abs(stfts) ** self.power

    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        self.features, num_spectrogram_bins, self.sample_rate, *self.freq_range
    )
    mel_spectrograms = magnitude_spectrograms @ mel_matrix

    def log_scale(x, floor, offset, scalar):
      """TensorFlow port of audio_utils.log_scale."""
      return scalar * tf.math.log(tf.maximum(x, floor) + offset)

    if isinstance(self.scaling_config, frontend.LogScalingConfig):
      # TODO(bartvm): Probably needs standardization step to stabilize training.
      features[self.name] = log_scale(
          mel_spectrograms, **dataclasses.asdict(self.scaling_config)
      )
    elif self.scaling_config is None:
      features[self.name] = mel_spectrograms
    else:
      raise ValueError('unknown scaling config')

    return features


@dataclasses.dataclass
class MFCC(FeaturesPreprocessOp):
  """Convert a spectrogram to MFC coefficients.

  This op assumes that the audio has already been processed into a log-magnitude
  mel-scale spectrogram.

  Attributes:
    num_coefficients: The number of MFC coefficients to retain.
    aggregate_over_time: If True, aggregates the MFC coefficients over time into
      four summary statistics: mean, standard deviation, min, and max, resulting
      in four feature vectors of shape `num_coefficients` that are then
      concatenated into a single feature vector. This mirrors the processing
      done in the BEANS benchmark (Hagiwara et al., 2022).
    name: The name of the feature to process.
  """

  num_coefficients: int
  aggregate_over_time: bool = True
  name: str = 'audio'

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    del dataset_info
    features = features.copy()
    features[self.name] = tf.signal.mfccs_from_log_mel_spectrograms(
        features[self.name]
    )[..., : self.num_coefficients]
    if self.aggregate_over_time:
      mean, variance = tf.nn.moments(features[self.name], axes=[-2])
      features[self.name] = tf.concat(
          [
              mean,
              tf.sqrt(variance),
              tf.reduce_min(features[self.name], axis=-2),
              tf.reduce_max(features[self.name], axis=-2),
          ],
          axis=-1,
      )

    return features


@dataclasses.dataclass
class LabelsToString(FeaturesPreprocessOp):
  """Converts labels to a string representation.

  Label values are joined using `separator`.

  Attributes:
    names: The labels to convert to a string representation.
    separator: The separator character to use.
  """

  names: tuple[str, ...] = ('label', 'genus', 'family', 'order', 'bg_labels')
  separator: str = ' '

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features = features.copy()
    for name in self.names:
      if name not in features:
        continue
      features[name] = tf.strings.reduce_join(
          tf.gather(
              tf.constant(dataset_info.features[name].feature.names),
              features[name],
          ),
          separator=self.separator,
      )

    return features


@dataclasses.dataclass
class LabelConversionConstants:
  """TF constants created while executing `ConvertBirdTaxonomyLabels`.

  Attributes:
    tables: a mapping from feature name to StaticHashTable for label conversion.
    masks: a mapping from feature name to mask for the translated labels.
  """

  tables: dict[str, tf.lookup.StaticHashTable]
  masks: dict[str, tf.Tensor]


@dataclasses.dataclass
class ConvertBirdTaxonomyLabels(FeaturesPreprocessOp):
  """Convert to a target set of classes and generate taxonomy labels."""

  source_namespace: str = 'ebird2021'
  target_class_list: str = 'ebird2021'
  species_feature_name: str = 'label'
  species_bg_label_name: str = 'bg_labels'
  add_taxonomic_labels: bool = True
  # Whether to add output features indicating which classes are represented
  # in the source dataset.
  output_masks: bool = True

  # The following members are for cached / stateful data.
  db: namespace_db.NamespaceDatabase | None = None

  def __post_init__(self):
    # Create NamespaceDatabase in post_init to avoid loading CSVs repeatedly.
    # Note that we purposefully avoid creating TF constants here. All TF
    # constant need to be created within the scope of `tf.data.Dataset.map`
    # (which in this case means inside __call__) so that the pipeline can be
    # applied multiple times on different datasets. Otherwise, in subsequent
    # pipeline applications TF will attempt to re-use previous constants
    # belonging to a different tf.function.
    self.db = namespace_db.NamespaceDatabase.load_csvs()

  def load_tables(
      self, source_class_list: namespace.ClassList
  ) -> LabelConversionConstants:
    """Construct TF StaticHashTables from namespace db info.

    Args:
      source_class_list: ClassList for the soruce dataset.

    Returns:
      TF constants needed for the execution of this preprocessing op.
    """
    tables = {}
    masks = {}
    target_classes = self.db.class_lists[self.target_class_list]

    label_table, label_mask = source_class_list.get_class_map_tf_lookup(
        target_classes
    )
    tables[self.species_feature_name] = label_table
    masks[self.species_feature_name] = label_mask
    tables[self.species_bg_label_name] = label_table
    masks[self.species_bg_label_name] = label_mask

    # Avoid searching for taxonomic mappings if `self.add_taxonomic_labels ==
    # False`, because it's possible that such a mapping doesn't exist.
    if self.add_taxonomic_labels:
      for key in ['genus', 'family', 'order']:
        # This is surprisingly tricky to get right for mismatched eval sets.
        # First map the source and target classes (eg, eval dataset species and
        # model ClassList) into the target namespace (eg, genera). This creates
        # two different ClassLists of genera. We then map the source genera to
        # the target genera to obtain an appropriate label_mask.
        namespace_mapping = self.db.mappings[
            self.source_namespace + '_to_' + key
        ]
        source_taxa_classes = source_class_list.apply_namespace_mapping(
            namespace_mapping
        )
        target_taxa_classes = target_classes.apply_namespace_mapping(
            namespace_mapping
        )
        namespace_table, _ = source_class_list.get_namespace_map_tf_lookup(
            namespace_mapping
        )
        class_table, label_mask = source_taxa_classes.get_class_map_tf_lookup(
            target_taxa_classes
        )
        tables[key + '_namespace'] = namespace_table
        tables[key + '_class'] = class_table
        masks[key] = label_mask

    return LabelConversionConstants(tables=tables, masks=masks)

  def convert_labels(
      self,
      features: Features,
      key: str,
      output_key: str,
      label_conversion_constants: LabelConversionConstants,
  ) -> Features:
    """Get a transformation for a given ClassList."""
    tables = label_conversion_constants.tables
    masks = label_conversion_constants.masks
    if output_key in (self.species_feature_name, self.species_bg_label_name):
      table = tables[key]
      label_mask = masks[key]
      output_labels = table.lookup(features[key])
    else:
      namespace_table = tables[output_key + '_namespace']
      class_table = tables[output_key + '_class']
      output_labels = class_table.lookup(namespace_table.lookup(features[key]))
      label_mask = masks[output_key]

    # Drop unknown labels.
    output_labels = tf.gather(output_labels, tf.where(output_labels >= 0)[:, 0])
    # Convert to MultiHot encoding.
    class_list_size = label_mask.shape[0]
    output_labels = tf.clip_by_value(
        tf.reduce_sum(
            tf.one_hot(output_labels, class_list_size, dtype=tf.int64), axis=0
        ),
        0,
        1,
    )
    return {output_key: output_labels, output_key + '_mask': label_mask}

  def convert_features(
      self, features: Features, source_classes: namespace.ClassList
  ) -> Features:
    """Convert features to target class list and add taxonomy labels."""
    output_features = features.copy()
    label_conversion_constants = self.load_tables(source_classes)

    output_features.update(
        self.convert_labels(
            features,
            self.species_feature_name,
            self.species_feature_name,
            label_conversion_constants,
        )
    )

    if self.species_bg_label_name in features:
      output_features.update(
          self.convert_labels(
              features,
              self.species_bg_label_name,
              self.species_bg_label_name,
              label_conversion_constants,
          )
      )

    if not self.add_taxonomic_labels:
      return output_features

    output_features.update(
        self.convert_labels(
            features,
            self.species_feature_name,
            'genus',
            label_conversion_constants,
        )
    )
    output_features.update(
        self.convert_labels(
            features,
            self.species_feature_name,
            'family',
            label_conversion_constants,
        )
    )
    output_features.update(
        self.convert_labels(
            features,
            self.species_feature_name,
            'order',
            label_conversion_constants,
        )
    )

    return output_features

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    source_classes = namespace.ClassList(
        'dataset',
        self.source_namespace,
        # TODO(vdumoulin): generalize this to labels beyond 'ignore'.
        # Some dataset variants (e.g. bird_taxonomy/downstream_slice_peaked)
        # use an 'ignore' label which is not part of the eBirds taxonomy. We
        # ignore this label; the mapping tables return an 'unknown' default
        # value, so all 'ignore' labels will naturally be converted to
        # 'unknown'.
        [
            n
            for n in dataset_info.features[self.species_feature_name].names
            if n != 'ignore'
        ],
    )
    output_features = self.convert_features(features, source_classes)
    return output_features


@dataclasses.dataclass
class OnlyJaxTypes(FeaturesPreprocessOp):
  """Discards tensors that are not supported by JAX (e.g., non-numeric).

  This must be done before batching.
  """

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    new_features = {}
    for name, feature in features.items():
      if (
          isinstance(feature, tf.Tensor)
          and hasattr(jnp, feature.dtype.name)
          or feature.dtype is tf.bool
      ):
        new_features[name] = feature
    return new_features


@dataclasses.dataclass
class OnlyKeep(FeaturesPreprocessOp):
  """Discards features with names not in `names`.

  Attributes:
    names: The names of features to keep.
  """

  names: Iterable[str]

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    return {
        name: feature
        for name, feature in features.items()
        if name in self.names
    }


@dataclasses.dataclass
class FilterMultiLabelRecordings(DatasetPreprocessOp):
  """Filters out recordings that have multiple foreground labels."""

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    def _predicate(features):
      return tf.math.equal(tf.shape(features['label'])[0], 1)

    return dataset.filter(_predicate)


@dataclasses.dataclass
class FilterByFeature(DatasetPreprocessOp):
  """Filters the dataset by feature values.

  Attributes:
    filtering_df_path: Path to a single-column, CSV-serialized DataFrame whose
      column name represents the feature name used for the filtering operation
      and whose rows contain the allowed feature values.
    complement: Whether to perform the complement of the filtering operation,
      i.e., swap which dataset elements are filtered and which are kept.
  """

  filtering_df_path: str
  complement: bool = False

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    df = pd.read_csv(self.filtering_df_path)

    if len(df.columns) != 1:
      raise ValueError(
          'filtering_df_path should point to a single-column DataFrame.'
      )

    (feature_name,) = df.columns
    feature_dtype = df[feature_name].dtype
    feature_values = df[feature_name].values

    feature_values_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(feature_values, dtype=feature_dtype),
            values=tf.range(len(feature_values), dtype=feature_dtype),
        ),
        default_value=-1,
    )

    def _predicate(features):
      value = tf.cast(features[feature_name], feature_dtype)
      should_include = feature_values_table.lookup(value) > -1
      if self.complement:
        should_include = ~should_include
      return should_include

    return dataset.filter(_predicate)


@dataclasses.dataclass
class HashId(FeaturesPreprocessOp):
  """Hashes a tfds_id into a unique integer."""

  num_buckets: int = int(1e9)

  def __call__(
      self, features: Features, dataset_info: tfds.core.DatasetInfo
  ) -> Features:
    features['tfds_id'] = tf.strings.to_hash_bucket_fast(
        features['tfds_id'], self.num_buckets
    )
    return features


@dataclasses.dataclass
class Shuffle(DatasetPreprocessOp):
  """Shuffles the dataset."""

  shuffle_buffer_size: int
  seed: int | None = None

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    return dataset.shuffle(self.shuffle_buffer_size, seed=self.seed)


@dataclasses.dataclass
class Repeat(DatasetPreprocessOp):
  """Repeats the data infinitely."""

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    return dataset.repeat()


@dataclasses.dataclass
class Batch(DatasetPreprocessOp):
  """Collects samples into batches.

  This preprocessing operation drops the remainder by default.

  Attributes:
    batch_size: The batch size to use.
    split_across_devices: If true, the minibatch will be split into smaller
      minibatches to be distributed across the local devices present. This is
      useful for distributed training.
    drop_remainder: Whether or not to drop remainder batch. Note that in the
      multi-device setting, examples will still be dropped if the dataset size
      is not a multiple of the batch size divided by the number of devices.
  """

  batch_size: int
  split_across_devices: bool = False
  drop_remainder: bool = True

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    if self.split_across_devices:
      if self.batch_size % jax.device_count():
        raise ValueError(
            f'batch size ({self.batch_size}) must be divisible by '
            f'number of devices ({jax.device_count()}).'
        )
      logging.info(
          'Splitting batch across %d devices, with local device count %d.',
          jax.device_count(),
          jax.local_device_count(),
      )
      dataset = dataset.batch(
          self.batch_size // jax.device_count(), drop_remainder=True
      )
      return dataset.batch(
          jax.local_device_count(), drop_remainder=self.drop_remainder
      )
    else:
      return dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)


@dataclasses.dataclass
class ExtractStridedWindows(DatasetPreprocessOp):
  """Extracts strided windows from examples.

  Attributes:
    window_length_sec: The window interval length to use, in seconds.
    window_stride_sec: The stride over which to slide the window.
    pad_end: Whether to pad the end of the recording. If True, window positions
      that are past the end of the recording are padded with zeros until the
      window moves fully past the end of the recording. Otherwise, only window
      positions that fully overlap the recording are considered.
  """

  window_length_sec: float
  window_stride_sec: float
  pad_end: bool = True

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    sample_rate = dataset_info.features['audio'].sample_rate
    window_length = int(sample_rate * self.window_length_sec)
    window_stride = int(sample_rate * self.window_stride_sec)

    def map_fn(example):
      example['audio'] = tf.signal.frame(
          signal=example['audio'],
          frame_length=window_length,
          frame_step=window_stride,
          pad_end=self.pad_end,
      )
      # At this point, example['audio'] has shape [num_windows, window_length].
      # We assign a unique sequential ID in [0, num_windows - 1] to each window.
      example['segment_id'] = tf.range(
          tf.shape(example['audio'])[0], dtype=tf.int64
      )
      example['segment_start'] = tf.cast(
          example['segment_id'] * window_stride, example['segment_start'].dtype
      )
      example['segment_end'] = tf.cast(
          example['segment_start'] + window_length, example['segment_end'].dtype
      )

      # Other features are shared across slices, so we repeat them across the
      # first axis.
      feature_names = ('audio', 'segment_id', 'segment_start', 'segment_end')
      for key, value in (
          (key, value)
          for key, value in example.items()
          if key not in feature_names
      ):
        value = tf.expand_dims(value, 0)
        value = tf.tile(
            value,
            [tf.shape(example['audio'])[0]] + [1] * (value.shape.ndims - 1),
        )
        example[key] = value
      return example

    # Unbatching yields slices one by one.
    return dataset.map(map_fn).unbatch()


@dataclasses.dataclass
class DenselyAnnotateWindows(DatasetPreprocessOp):
  """Densely annotates sliding windows of the dataset's 'audio'.

  After extracting slided windows on the dataset's 'audio' feature, this
  preprocessing distributes the labels corresponding to each annotated segment
  to all windows that intersect in time within a given threshold. Each window is
  assigned all labels that are included within each overlapping annotation and
  the 'annotation_start' and 'annotation_end' features. In the case where a
  given window overlaps with more than one annotation, that window is assigned
  the labels of each annotation.

  Process: compare each 'audio' window's 'segment_start' and 'segment_end' times
  with the time delimiters in its 'annotation_start' and 'annotation_end'; if
  there is an absolute overlap of at least `overlap_threshold_sec` with the
  segment bounds, the window receives the segment labels.

  Attributes:
    overlap_threshold_sec: The minimum overlap, in seconds, between a window and
      a labeled segment for the former to inherit its label. This overlap is
      translated into a number of audio samples using the dataset's sampling
      rate. If None, we set the threshold to one audio sample.
    drop_annotation_bounds: If True, remove the 'annotation_start' and
      'annotation_end' features. If False, the annotation bound features are
      placed in an array of size [num_labels], with zeros for entries where no
      label is present. This allows downstream batching, since the features are
      of fixed size. (We also add features for annotation_size and
      intersection_size for downstream debugging and analysis.)
  """

  overlap_threshold_sec: float | None = None
  drop_annotation_bounds: bool = False

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    sample_rate = dataset_info.features['audio'].sample_rate
    overlap_threshold = (
        1
        if self.overlap_threshold_sec is None
        else int(sample_rate * self.overlap_threshold_sec)
    )

    def map_fn(example):
      example = example.copy()

      # A window and an annotated segment overlaps (by at least
      # `overlap_threshold`) if the following is true:
      #     max(segment_start, annotation_start)
      #       <= min(segment_end, annotation_end) - overlap_threshold
      # Note that `example['segment_{start|end}']` is uint64-valued and
      # `example['annotation_{start|end}']` is a variable-length sequence of
      # integers and the operation is broadcasted across all segments.

      # Find the start and end of he intersection of the annotation and segment.
      # If inter_end < inter_start, the intersection is empty.
      inter_end = tf.cast(
          tf.minimum(example['segment_end'], example['annotation_end']),
          tf.int64,
      )
      inter_start = tf.cast(
          tf.maximum(example['segment_start'], example['annotation_start']),
          tf.int64,
      )
      overlap_comparison = tf.cast(
          inter_end - inter_start - overlap_threshold >= 0, tf.bool
      )
      overlap_indices = tf.reshape(tf.where(overlap_comparison), [-1])

      if self.drop_annotation_bounds:
        del example['annotation_start']
        del example['annotation_end']
      else:
        # Add per-label annotation metadata. When a label is not present, these
        # data default to zero.
        # Note: In case a segment has multiple annotations for the same species,
        # only one annotation will be described by these metadata.
        num_classes = len(dataset_info.features['label'].names)
        label_idxs = tf.gather(example['label'], overlap_indices)
        example['intersection_size'] = tf.maximum(inter_end - inter_start, 0)
        example['annotation_length'] = tf.cast(
            example['annotation_end'], tf.int64
        ) - tf.cast(example['annotation_start'], tf.int64)

        for k in (
            'annotation_start',
            'annotation_end',
            'intersection_size',
            'annotation_length',
        ):
          example[k] = tf.cast(tf.gather(example[k], overlap_indices), tf.int64)
          example[k] = tf.scatter_nd(
              indices=label_idxs[:, tf.newaxis],
              updates=example[k],
              shape=[num_classes],
          )

      example['label'] = tf.gather(example['label'], overlap_indices)
      return example

    return dataset.map(map_fn)


@dataclasses.dataclass
class Cache(DatasetPreprocessOp):
  """Caches the dataset.

  Attributes:
    filename: Where to cache the dataset. If left empty, the dataset is cached
      in memory.
  """

  filename: str = ''

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    del dataset_info
    return dataset.cache(filename=self.filename)


@dataclasses.dataclass
class FilterDropLabel(DatasetPreprocessOp):
  """Drop any examples with the target label."""

  target_label: str = 'unknown'

  def __call__(
      self, dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo
  ) -> tf.data.Dataset:
    label_names = dataset_info.features['label'].names
    if self.target_label not in label_names:
      return dataset

    filter_idx = label_names.index(self.target_label)

    def _pred(features):
      return tf.math.logical_not(tf.reduce_any(filter_idx == features['label']))

    return dataset.filter(_pred)
