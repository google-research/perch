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
from unittest import mock

from chirp.data import pipeline
from chirp.models import frontend
from chirp.taxonomy import namespace_db
from chirp.tests import fake_dataset
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class PipelineTest(parameterized.TestCase):

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
        'audio': tf.random.uniform([2, 100], dtype=tf.float32),
        'segment_start': tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'segment_end': tf.convert_to_tensor([117, 164], dtype=tf.int64),
        'label': tf.convert_to_tensor([[1], [2]], dtype=tf.int64),
        'label_str': tf.convert_to_tensor(
            [['placeholder'], ['placeholder']], dtype=tf.string
        ),
        'bg_labels': tf.convert_to_tensor([[2, 3], [4, 5]], dtype=tf.int64),
        'filename': tf.convert_to_tensor(
            ['placeholder', 'placeholder'], dtype=tf.string
        ),
    }
    ds = tf.data.Dataset.from_tensor_slices(examples)
    ds = pipeline.Pipeline(
        [pipeline.OnlyJaxTypes(), pipeline.MultiHot()], deterministic=True
    )(ds, self._builder.info)
    mixed_ds = pipeline.Pipeline([pipeline.MixAudio(1.0)], deterministic=True)(
        ds, self._builder.info
    )
    mixed_example = next(mixed_ds.as_numpy_iterator())
    np.testing.assert_allclose(
        mixed_example['audio'], examples['audio'][0] + examples['audio'][1]
    )

    np.testing.assert_equal(
        mixed_example['bg_labels'],
        np.asarray(
            [0, 0, 1, 1, 1, 1]
            + [0] * (self._builder.info.features['bg_labels'].num_classes - 6),
            dtype=np.int32,
        ),
    )

    unmixed_ds = pipeline.Pipeline(
        [pipeline.MixAudio(mixin_prob=0.0)], deterministic=True
    )(ds, self._builder.info)
    for x, y in tf.data.Dataset.zip((ds, unmixed_ds)).as_numpy_iterator():
      for key in x:
        if key in ('source_audio', 'segment_start', 'segment_end'):
          np.testing.assert_equal(x[key], y[key][:1])
          np.testing.assert_equal(np.zeros_like(x[key]), y[key][1:])
        else:
          np.testing.assert_equal(x[key], y[key], err_msg=f'{key} not equal')

  def test_process_example(self):
    sample_rate_hz = self._builder.info.features['audio'].sample_rate
    audio_length_s = 6
    audio_length_samples = sample_rate_hz * audio_length_s
    input_gain = 10.0
    window_size_s = 5
    min_gain = 0.15
    max_gain = 0.25

    example = {
        'audio': tf.random.uniform(
            [audio_length_samples],
            minval=-input_gain,
            maxval=input_gain,
            dtype=tf.float32,
        ),
        'segment_start': tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'segment_end': tf.convert_to_tensor(
            [17 + audio_length_samples, 64 + audio_length_samples],
            dtype=tf.int64,
        ),
        'label': tf.convert_to_tensor([1], dtype=tf.int64),
        'label_str': tf.convert_to_tensor(['placeholder'], dtype=tf.string),
        'bg_labels': tf.convert_to_tensor([2, 3], dtype=tf.int64),
        'filename': tf.convert_to_tensor('placeholder', dtype=tf.string),
    }
    example = pipeline.OnlyJaxTypes()(example, self._builder.info)
    example = pipeline.MultiHot()(example, self._builder.info)

    # The bg_labels feature should be multi-hot encoded.
    num_classes = self._builder.info.features['bg_labels'].feature.num_classes
    np.testing.assert_equal(
        example['bg_labels'].numpy(),
        np.asarray([0, 0, 1, 1] + [0] * (num_classes - 4), dtype=np.int32),
    )

    example = pipeline.RandomSlice(window_size_s, names=('audio',))(
        example, self._builder.info
    )
    example = pipeline.RandomNormalizeAudio(
        min_gain, max_gain, names=('audio',)
    )(example, self._builder.info)

    # The audio feature should be trimmed to the requested length, and its
    # maximum absolute value should be within [min_gain, max_gain].
    audio = example['audio'].numpy()
    self.assertEqual(audio.shape, (sample_rate_hz * window_size_s,))
    # There is a constant value of 0.01 added to the denominator during
    # normalization.
    self.assertTrue(
        input_gain / (input_gain + 0.01) * min_gain
        <= np.abs(audio).max()
        <= input_gain / (input_gain + 0.01) * max_gain
    )

    # The label feature should be one-hot encoded.
    key = 'label'
    np.testing.assert_equal(
        example[key].numpy(),
        np.asarray(
            [0, 1, 0]
            + [0] * (self._builder.info.features[key].num_classes - 3),
            dtype=np.int32,
        ),
    )

    # The label_str and filename features should be deleted.
    for key in ('label_str', 'filename'):
      self.assertNotIn(key, example)

  def test_get_dataset(self):
    test_pipeline = pipeline.Pipeline([
        pipeline.OnlyJaxTypes(),
        pipeline.MultiHot(),
        pipeline.MixAudio(mixin_prob=0.25),
        pipeline.Batch(8),
        pipeline.RandomSlice(window_size=5),
        pipeline.RandomNormalizeAudio(min_gain=0.15, max_gain=0.25),
    ])
    for split in self._builder.info.splits.values():
      dataset, _ = pipeline.get_dataset(
          split.name,
          dataset_directory=self._builder.data_dir,
          pipeline=test_pipeline,
      )

      example = next(dataset.as_numpy_iterator())
      self.assertLen(example['audio'].shape, 2)
      self.assertLen(example['source_audio'].shape, 3)
      self.assertSetEqual(
          set(example.keys()),
          {
              'audio',
              'source_audio',
              'bg_labels',
              'label',
              'segment_start',
              'segment_end',
              'recording_id',
              'segment_id',
          },
      )
      # Check error raising when getting last dataset split without a pipeline.
      with self.assertRaises(ValueError):
        pipeline.get_dataset(
            split.name, dataset_directory=self._builder.data_dir
        )

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
        'label': tf.constant([0, 20, 40, 78, 79, 10655, 10932, -1], tf.int64),
        'bg_labels': tf.constant([18, 1000], tf.int64),
    }
    converter = pipeline.ConvertBirdTaxonomyLabels(
        target_class_list='xenocanto'
    )
    converted = converter.convert_features(example, source_class_set)
    for name, shape, num in (
        ('label', 10932, 4),
        ('bg_labels', 10932, 1),
        ('genus', 2333, 4),
        ('family', 249, 3),
        ('order', 41, 2),
    ):
      print(name, shape, num, sum(converted[name].numpy()))
      self.assertIn(name, converted)
      self.assertLen(converted[name].shape, 1)
      self.assertEqual(converted[name].shape[0], shape)
      self.assertEqual(
          converted[name].shape[0], converted[name + '_mask'].shape[0]
      )
      self.assertEqual(sum(converted[name].numpy()), num)

    for image_name, image_size in (
        ('label_mask', 79),
        ('genus_mask', 62),
        ('family_mask', 30),
        ('order_mask', 11),
    ):
      self.assertIn(image_name, converted)
      self.assertLen(converted[image_name].shape, 1)
      self.assertEqual(np.sum(converted[image_name].numpy()), image_size)

  def test_labels_to_string(self):
    examples = {
        'segment_start': tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'label': tf.convert_to_tensor([[1], [2]], dtype=tf.int64),
        'bg_labels': tf.convert_to_tensor([[2, 3], [4, 5]], dtype=tf.int64),
        'filename': tf.convert_to_tensor(
            ['placeholder', 'placeholder'], dtype=tf.string
        ),
    }
    ds = tf.data.Dataset.from_tensor_slices(examples)
    ds = pipeline.Pipeline(
        [
            pipeline.LabelsToString(),
        ]
    )(
        ds, self._builder.info
    ).batch(2)
    class_names = self._builder.info.features['label'].feature.names
    processed_example = next(ds.as_numpy_iterator())
    np.testing.assert_equal(
        processed_example['segment_start'], examples['segment_start']
    )
    np.testing.assert_equal(
        processed_example['label'],
        [class_names[1].encode('utf-8'), class_names[2].encode('utf-8')],
    )
    np.testing.assert_equal(
        processed_example['bg_labels'],
        [
            f'{class_names[2]} {class_names[3]}'.encode('utf-8'),
            f'{class_names[4]} {class_names[5]}'.encode('utf-8'),
        ],
    )
    np.testing.assert_equal(processed_example['filename'], examples['filename'])

  def test_only_keep(self):
    examples = {
        'segment_start': tf.convert_to_tensor([17, 64], dtype=tf.int64),
        'label': tf.convert_to_tensor([[1], [2]], dtype=tf.int64),
        'bg_labels': tf.convert_to_tensor([[2, 3], [4, 5]], dtype=tf.int64),
        'filename': tf.convert_to_tensor(
            ['placeholder', 'placeholder'], dtype=tf.string
        ),
    }
    ds = tf.data.Dataset.from_tensor_slices(examples)
    ds = pipeline.Pipeline(
        [
            pipeline.OnlyKeep(names=['segment_start', 'bg_labels']),
        ]
    )(ds, self._builder.info).batch(2)
    processed_example = next(ds.as_numpy_iterator())
    self.assertSameElements(
        processed_example.keys(), ['segment_start', 'bg_labels']
    )
    np.testing.assert_equal(
        processed_example['segment_start'], examples['segment_start']
    )
    np.testing.assert_equal(
        processed_example['bg_labels'], examples['bg_labels']
    )

  @parameterized.parameters(
      None,
      frontend.LogScalingConfig(floor=1e-5, scalar=0.1),
  )
  def test_melspec(self, scaling_config):
    batch_size = 3
    sample_rate_hz = 22050

    time_size = 5 * sample_rate_hz
    audio = tf.math.sin(tf.linspace(0.0, 440 * jnp.pi, time_size))
    noise = 0.01 * tf.random.normal((batch_size, time_size))
    signal = audio + noise

    model = frontend.MelSpectrogram(
        features=160,
        stride=sample_rate_hz // 100,
        kernel_size=512,  # ~0.08 * 32,000
        sample_rate=sample_rate_hz,
        freq_range=(60, 10_000),
        scaling_config=scaling_config,
    )
    melspec = model.apply({}, jnp.array(signal))

    melspec_tf = pipeline.MelSpectrogram(
        features=160,
        stride=sample_rate_hz // 100,
        kernel_size=512,  # ~0.08 * 32,000
        sample_rate=sample_rate_hz,
        freq_range=(60, 10_000),
        scaling_config=scaling_config,
    )({'audio': signal}, dataset_info=None)['audio']

    np.testing.assert_allclose(melspec, melspec_tf.numpy(), atol=1e-5)

  @parameterized.named_parameters(('pad_end', True), ('no_pad_end', False))
  def test_extract_strided_slices(self, pad_end):
    sample_rate = self._builder.info.features['audio'].sample_rate
    length_sec = 5
    stride_sec = 2.5
    length = int(length_sec * sample_rate)
    stride = int(stride_sec * sample_rate)

    original_dataset = self._builder.as_dataset('train')
    original_examples = next(
        original_dataset.batch(len(original_dataset)).as_numpy_iterator()
    )
    dataset = pipeline.ExtractStridedWindows(
        window_length_sec=length_sec,
        window_stride_sec=stride_sec,
        pad_end=pad_end,
    )(original_dataset, self._builder.info)
    examples = next(dataset.batch(len(dataset)).as_numpy_iterator())

    # The fake_dataset builder creates 6s recordings. This results in one full
    # slice and two zero-padded slices when using a 5s window with stride 2.5s.
    # We expect one slice per example if padding='VALID' and three slices per
    # example otherwise.
    self.assertLen(
        dataset, len(original_dataset) * 3 if pad_end else len(original_dataset)
    )

    # Verify slices have the expected length.
    self.assertEqual(examples['audio'].shape[1], length)

    # The segment start and end indices should reflect the window sliding over
    # the audio.
    np.testing.assert_equal(
        examples['segment_start'],
        [0, stride, 2 * stride] * len(original_dataset) if pad_end else 0,
    )
    np.testing.assert_equal(
        examples['segment_end'],
        [length, length + stride, length + 2 * stride] * len(original_dataset)
        if pad_end
        else length,
    )
    # The segment IDs should reflect the sliding window's position.
    np.testing.assert_equal(
        examples['segment_id'],
        [0, 1, 2] * len(original_dataset) if pad_end else 0,
    )
    # The other features should be replicated across slices.
    other_feature_names = [
        k
        for k in original_examples
        if k not in ('audio', 'segment_start', 'segment_end', 'segment_id')
    ]
    for key in other_feature_names:
      np.testing.assert_equal(
          examples[key],
          np.repeat(original_examples[key], 3, axis=0)
          if pad_end
          else original_examples[key],
      )
    # With a recording length of 6s, a window size of 5s a window stride of
    # 2.5s, and with end-padding , we expect the slices to cycle between a full
    # slice, a slice with 1.5s of zero padding, and a slice with 4s of zero
    # padding.
    if pad_end:
      np.testing.assert_equal(
          examples['audio'][1::3, -int(1.5 * sample_rate) :], 0
      )
      np.testing.assert_equal(
          examples['audio'][2::3, -int(4.0 * sample_rate) :], 0
      )

  def test_densely_annotate_windows_no_overlap_threshold(self):
    # Sampling rate is 10, so divide the timestamps by 10 for seconds.
    original_example = {
        'segment_start': np.array(10, dtype=np.int64),
        'segment_end': np.array(50, dtype=np.int64),
        'annotation_start': np.array([10, 30, 45], dtype=np.int64),
        'annotation_end': np.array([20, 60, 90], dtype=np.int64),
        'label': np.array([0, 1, 2], dtype=np.int64),
    }
    fake_dataset_info = mock.MagicMock(
        features={'audio': mock.MagicMock(sample_rate=10)}
    )
    original_dataset = tf.data.Dataset.from_tensors(original_example)
    annotated_dataset = pipeline.DenselyAnnotateWindows(
        overlap_threshold_sec=0
    )(original_dataset, fake_dataset_info)
    annotated_dataset = next(annotated_dataset.as_numpy_iterator())

    expected_dataset = {
        'segment_start': np.array(10, dtype=np.int64),
        'segment_end': np.array(50, dtype=np.int64),
        'annotation_start': np.array([10, 30, 45], dtype=np.int64),
        'annotation_end': np.array([20, 60, 90], dtype=np.int64),
        'label': np.array([0, 1, 2], dtype=np.int64),
    }

    for key, expected_value in expected_dataset.items():
      np.testing.assert_equal(expected_value, annotated_dataset[key])

  def test_densely_annotate_windows_overlap_1sec(self):
    # Sampling rate is 10, so divide the timestamps by 10 for seconds.
    original_example = {
        'segment_start': np.array(10, dtype=np.int64),
        'segment_end': np.array(50, dtype=np.int64),
        'annotation_start': np.array([10, 30, 45], dtype=np.int64),
        'annotation_end': np.array([20, 60, 90], dtype=np.int64),
        'label': np.array([0, 1, 2], dtype=np.int64),
    }
    fake_dataset_info = mock.MagicMock(
        features={'audio': mock.MagicMock(sample_rate=10)}
    )
    original_dataset = tf.data.Dataset.from_tensors(original_example)
    annotated_dataset = pipeline.DenselyAnnotateWindows(
        overlap_threshold_sec=1
    )(original_dataset, fake_dataset_info)
    annotated_dataset = next(annotated_dataset.as_numpy_iterator())

    expected_dataset = {
        'segment_start': np.array(10, dtype=np.int64),
        'segment_end': np.array(50, dtype=np.int64),
        'annotation_start': np.array([10, 30], dtype=np.int64),
        'annotation_end': np.array([20, 60], dtype=np.int64),
        'label': np.array([0, 1], dtype=np.int64),
    }

    for key, expected_value in expected_dataset.items():
      np.testing.assert_equal(expected_value, annotated_dataset[key])


if __name__ == '__main__':
  absltest.main()
