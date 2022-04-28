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

"""Tests for pipeline."""

from chirp.data import pipeline
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds


class FakeDataset(tfds.core.GeneratorBasedBuilder):
  """Fake dataset."""

  VERSION = tfds.core.Version('1.0.0')

  LABEL_NAMES = [str(i) for i in range(90)]
  GENUS_NAMES = [str(i) for i in range(60)]
  FAMILY_NAMES = [str(i) for i in range(30)]
  ORDER_NAMES = [str(i) for i in range(20)]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'audio':
                tfds.features.Audio(dtype=tf.float32, sample_rate=44_100),
            'label':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.LABEL_NAMES)),
            'label_str':
                tfds.features.Sequence(tfds.features.Text()),
            'bg_labels':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.LABEL_NAMES)),
            'genus':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.GENUS_NAMES)),
            'family':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.FAMILY_NAMES)),
            'order':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=self.ORDER_NAMES)),
            'filename':
                tfds.features.Text(),
        }),
        description='Fake dataset.',
    )

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(100),
        'test': self._generate_examples(20),
    }

  def _generate_examples(self, num_examples):
    for i in range(num_examples):
      yield i, {
          'audio': np.random.uniform(-1.0, 1.0, [44_100]),
          'label': np.random.choice(self.LABEL_NAMES, size=[3]),
          'label_str': ['placeholder'] * 3,
          'bg_labels': np.random.choice(self.LABEL_NAMES, size=[2]),
          'genus': np.random.choice(self.GENUS_NAMES, size=[1]),
          'family': np.random.choice(self.FAMILY_NAMES, size=[1]),
          'order': np.random.choice(self.ORDER_NAMES, size=[2]),
          'filename': 'placeholder',
      }


@pytest.fixture(scope='session')
def builder(tmpdir_factory):
  data_dir = tmpdir_factory.mktemp('data_dir')
  fake_builder = FakeDataset(data_dir=data_dir)
  fake_builder.download_and_prepare()
  return fake_builder


def test_process_example(builder):  # pylint: disable=redefined-outer-name
  sample_rate_hz = builder.info.features['audio'].sample_rate
  audio_length_s = 6
  input_gain = 10.0
  window_size_s = 5
  min_gain = 0.15
  max_gain = 0.25

  example = {
      'audio':
          tf.random.uniform([sample_rate_hz * audio_length_s],
                            minval=-input_gain,
                            maxval=input_gain,
                            dtype=tf.float32),
      'label':
          tf.convert_to_tensor([1], dtype=tf.int64),
      'label_str':
          tf.convert_to_tensor(['placeholder'], dtype=tf.string),
      'bg_labels':
          tf.convert_to_tensor([2, 3], dtype=tf.int64),
      'genus':
          tf.convert_to_tensor([1], dtype=tf.int64),
      'family':
          tf.convert_to_tensor([1], dtype=tf.int64),
      'order':
          tf.convert_to_tensor([1], dtype=tf.int64),
      'filename':
          tf.convert_to_tensor('placeholder', dtype=tf.string),
  }

  example = pipeline.multi_hot(example=example, info=builder.info)

  # The bg_labels feature should be multi-hot encoded.
  num_classes = builder.info.features['bg_labels'].feature.num_classes
  np.testing.assert_equal(
      example['bg_labels'].numpy(),
      np.asarray([0, 0, 1, 1] + [0] * (num_classes - 4), dtype=np.int32))

  example = pipeline.process_audio(
      example=example,
      info=builder.info,
      window_size_s=window_size_s,
      min_gain=min_gain,
      max_gain=max_gain)

  # The audio feature should be trimmed to the requested length, and its
  # maximum absolute value should be within [min_gain, max_gain].
  audio = example['audio'].numpy()
  assert audio.shape == (sample_rate_hz * window_size_s,)
  # There is a constant value of 0.01 added to the denominator during
  # normalization.
  assert (input_gain /
          (input_gain + 0.01) * min_gain <= np.abs(audio).max() <= input_gain /
          (input_gain + 0.01) * max_gain)

  # The label, genus, family, and order features should be one-hot encoded.
  for key in ('label', 'genus', 'family', 'order'):
    np.testing.assert_equal(
        example[key].numpy(),
        np.asarray(
            [0, 1, 0] + [0] * (builder.info.features[key].num_classes - 3),
            dtype=np.int32))

  # The label_str and filename features should be deleted.
  for key in ('label_str', 'filename'):
    assert key not in example


def test_get_dataset(builder):  # pylint: disable=redefined-outer-name
  batch_size = 4

  window_size_s = 5
  min_gain = 0.15
  max_gain = 0.25

  for split in builder.info.splits.values():
    dataset, _ = pipeline.get_dataset(
        split.name,
        dataset_directory=builder.data_dir,
        batch_size=batch_size,
        window_size_s=window_size_s,
        min_gain=min_gain,
        max_gain=max_gain)

    example = next(dataset.as_numpy_iterator())
    assert len(example['audio'].shape) == 2
    assert example['audio'].shape[0] == batch_size
    assert set(example.keys()) == {
        'audio', 'bg_labels', 'family', 'genus', 'label', 'order'
    }
