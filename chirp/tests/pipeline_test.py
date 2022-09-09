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
import os
import tempfile

from chirp.data import pipeline
from chirp.taxonomy import namespace_db
from chirp.tests import fake_dataset
import numpy as np
import tensorflow as tf

from absl.testing import absltest


class PipelineTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Test with two CPU devices.
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
    data_dir = tempfile.TemporaryDirectory('data_dir').name
    fake_builder = fake_dataset.FakeDataset(data_dir=data_dir)
    fake_builder.download_and_prepare()
    cls._builder = fake_builder

  def test_mixin(self):
    examples = {
        'audio':
            tf.random.uniform([2, 100], dtype=tf.float32),
        'segment_start':
            tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'segment_end':
            tf.convert_to_tensor([117, 164], dtype=tf.int64),
        'label':
            tf.convert_to_tensor([[1], [2]], dtype=tf.int64),
        'label_str':
            tf.convert_to_tensor([['placeholder'], ['placeholder']],
                                 dtype=tf.string),
        'bg_labels':
            tf.convert_to_tensor([[2, 3], [4, 5]], dtype=tf.int64),
        'filename':
            tf.convert_to_tensor(['placeholder', 'placeholder'],
                                 dtype=tf.string),
    }
    ds = tf.data.Dataset.from_tensor_slices(examples)
    ds = pipeline.Pipeline([
        pipeline.OnlyJaxTypes(),
        pipeline.MultiHot(),
    ])(ds, self._builder.info)
    mixed_ds = pipeline.Pipeline([
        pipeline.MixAudio(1.0),
    ])(ds, self._builder.info)
    mixed_example = next(mixed_ds.as_numpy_iterator())
    np.testing.assert_allclose(mixed_example['audio'],
                               examples['audio'][0] + examples['audio'][1])

    np.testing.assert_equal(
        mixed_example['bg_labels'],
        np.asarray(
            [0, 0, 1, 1, 1, 1] + [0] *
            (self._builder.info.features['bg_labels'].num_classes - 6),
            dtype=np.int32))

    unmixed_ds = pipeline.Pipeline([
        pipeline.MixAudio(mixin_prob=0.0),
    ])(ds, self._builder.info)
    for x, y in tf.data.Dataset.zip((ds, unmixed_ds)).as_numpy_iterator():
      for key in x:
        if key in ('source_audio', 'segment_start', 'segment_end'):
          np.testing.assert_equal(x[key], y[key][:1])
          np.testing.assert_equal(np.zeros_like(x[key]), y[key][1:])
        else:
          np.testing.assert_equal(x[key], y[key])

  def test_process_example(self):
    sample_rate_hz = self._builder.info.features['audio'].sample_rate
    audio_length_s = 6
    audio_length_samples = sample_rate_hz * audio_length_s
    input_gain = 10.0
    window_size_s = 5
    min_gain = 0.15
    max_gain = 0.25

    example = {
        'audio':
            tf.random.uniform([audio_length_samples],
                              minval=-input_gain,
                              maxval=input_gain,
                              dtype=tf.float32),
        'segment_start':
            tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'segment_end':
            tf.convert_to_tensor(
                [17 + audio_length_samples, 64 + audio_length_samples],
                dtype=tf.int64),
        'label':
            tf.convert_to_tensor([1], dtype=tf.int64),
        'label_str':
            tf.convert_to_tensor(['placeholder'], dtype=tf.string),
        'bg_labels':
            tf.convert_to_tensor([2, 3], dtype=tf.int64),
        'filename':
            tf.convert_to_tensor('placeholder', dtype=tf.string),
    }
    example = pipeline.OnlyJaxTypes()(example, self._builder.info)
    example = pipeline.MultiHot()(example, self._builder.info)

    # The bg_labels feature should be multi-hot encoded.
    num_classes = self._builder.info.features['bg_labels'].feature.num_classes
    np.testing.assert_equal(
        example['bg_labels'].numpy(),
        np.asarray([0, 0, 1, 1] + [0] * (num_classes - 4), dtype=np.int32))

    example = pipeline.RandomSlice(
        window_size_s, names=('audio',))(example, self._builder.info)
    example = pipeline.RandomNormalizeAudio(
        min_gain, max_gain, names=('audio',))(example, self._builder.info)

    # The audio feature should be trimmed to the requested length, and its
    # maximum absolute value should be within [min_gain, max_gain].
    audio = example['audio'].numpy()
    self.assertEqual(audio.shape, (sample_rate_hz * window_size_s,))
    # There is a constant value of 0.01 added to the denominator during
    # normalization.
    self.assertTrue(
        input_gain /
        (input_gain + 0.01) * min_gain <= np.abs(audio).max() <= input_gain /
        (input_gain + 0.01) * max_gain)

    # The label feature should be one-hot encoded.
    key = 'label'
    np.testing.assert_equal(
        example[key].numpy(),
        np.asarray(
            [0, 1, 0] + [0] *
            (self._builder.info.features[key].num_classes - 3),
            dtype=np.int32))

    # The label_str and filename features should be deleted.
    for key in ('label_str', 'filename'):
      self.assertNotIn(key, example)

  def test_get_dataset(self):

    for split in self._builder.info.splits.values():
      dataset, _ = pipeline.get_dataset(
          split.name, dataset_directory=self._builder.data_dir)

      example = next(dataset.as_numpy_iterator())
      self.assertLen(example['audio'].shape, 2)
      self.assertLen(example['source_audio'].shape, 3)
      self.assertSetEqual(
          set(example.keys()), {
              'audio', 'source_audio', 'bg_labels', 'label', 'segment_start',
              'segment_end'
          })

  def test_convert_bird_taxonomy_labels(self):
    db = namespace_db.NamespaceDatabase.load_csvs()
    source_class_set = db.class_lists['caples']
    target_class_set = db.class_lists['xenocanto']
    self.assertEqual(source_class_set.size, 79)
    self.assertEqual(target_class_set.size, 10932)

    # example labels include three 'good' labels and many out of range labels.
    # Good classes are 'amedip', 'comnig', 'macwar', and 'yerwar'.
    # The following table lists their index in the source_class_list,
    # label, genus, family, and order.
    # 0  amedip cinclus    cinclidae     passeriformes
    # 20 comnig chordeiles caprimulgidae caprimulgiformes
    # 40 macwar geothlypis parulidae     passeriformes
    # 78 yerwar setophaga  parulidae     passeriformes
    example = {
        'label': tf.constant([0, 20, 40, 78, 79, 10931, 10932, -1], tf.int64),
        'bg_labels': tf.constant([18, 1000], tf.int64),
    }
    converter = pipeline.ConvertBirdTaxonomyLabels(
        target_class_list='xenocanto')
    converted = converter.convert_features(example, source_class_set)
    for name, shape, num in (('label', 10932, 4), ('bg_labels', 10932, 1),
                             ('genus', 2333, 4), ('family', 249, 3), ('order',
                                                                      41, 2)):
      print(name, shape, num, sum(converted[name].numpy()))
      self.assertIn(name, converted)
      self.assertLen(converted[name].shape, 1)
      self.assertEqual(converted[name].shape[0], shape)
      self.assertEqual(converted[name].shape[0],
                       converted[name + '_mask'].shape[0])
      self.assertEqual(sum(converted[name].numpy()), num)

    for image_name, image_size in (('label_mask', 79), ('genus_mask', 62),
                                   ('family_mask', 30), ('order_mask', 11)):
      self.assertIn(image_name, converted)
      self.assertLen(converted[image_name].shape, 1)
      self.assertEqual(np.sum(converted[image_name].numpy()), image_size)


if __name__ == '__main__':
  absltest.main()
