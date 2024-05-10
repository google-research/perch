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

"""Tests for pipeline."""
import os
import tempfile
from unittest import mock

from chirp.data import utils as data_utils
from chirp.models import frontend
from chirp.preprocessing import pipeline
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from chirp.train_tests import fake_dataset
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
      dataset, _ = data_utils.get_dataset(
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
        data_utils.get_dataset(
            split.name, dataset_directory=self._builder.data_dir
        )

  def test_convert_bird_taxonomy_labels(self):
    db = namespace_db.load_db()
    np.random.seed(42)
    source_class_list = db.class_lists['caples']
    # Create a shuffled version of the source class list.
    source_classes = list(source_class_list.classes)
    np.random.shuffle(source_classes)
    source_class_list = namespace.ClassList('ebird2021', source_classes)
    target_class_list = db.class_lists['xenocanto']
    self.assertLen(source_class_list.classes, 79)
    self.assertLen(target_class_list.classes, 10932)

    # example labels include three 'good' labels and many out of range labels.
    # Good classes are 'amedip', 'comnig', 'macwar', and 'yerwar'.
    # The following table lists their index in the source_class_list,
    # label, genus, family, and order.
    # 0  amedip cinclus    cinclidae     passeriformes
    # 20 comnig chordeiles caprimulgidae caprimulgiformes
    # 40 macwar geothlypis parulidae     passeriformes
    # 78 yerwar setophaga  parulidae     passeriformes
    example = {
        'label': tf.constant(
            [
                source_class_list.classes.index('amedip'),
                source_class_list.classes.index('comnig'),
                source_class_list.classes.index('macwar'),
                source_class_list.classes.index('yerwar'),
                79,
                10655,
                10932,
                -1,
            ],
            tf.int64,
        ),
        'bg_labels': tf.constant([18, 1000], tf.int64),
    }
    converter = pipeline.ConvertBirdTaxonomyLabels(
        target_class_list='xenocanto'
    )
    converted = converter.convert_features(example, source_class_list)
    # Check species labels are correct.
    for species in ('amedip', 'comnig', 'macwar', 'yerwar'):
      target_idx = target_class_list.classes.index(species)
      self.assertEqual(converted['label'][target_idx], 1)

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

  def test_resample_audio(self):
    original_dataset = self._builder.as_dataset('train')
    original_examples = next(
        original_dataset.batch(len(original_dataset)).as_numpy_iterator()
    )
    # Six seconds at 32kHz, gives 192000 samples.
    original_length = original_examples['audio'].shape[1]
    original_sample_rate = self._builder.info.features['audio'].sample_rate

    resampled_examples = pipeline.ResampleAudio(target_sample_rate=16000)(
        original_examples, self._builder.info
    )
    expected_length = int(16000 * original_length / original_sample_rate)
    self.assertEqual(
        resampled_examples['audio'].shape[0],
        original_examples['audio'].shape[0],
    )
    self.assertEqual(resampled_examples['audio'].shape[1], expected_length)
    self.assertLen(resampled_examples['audio'].shape, 2)

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
        features={
            'audio': mock.MagicMock(sample_rate=10),
            'label': mock.MagicMock(names=('dowwoo', 'daejun', 'pilwoo')),
        }
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
        'segment_start': np.array(10, dtype=np.uint64),
        'segment_end': np.array(50, dtype=np.uint64),
        'annotation_start': np.array([10, 30, 45], dtype=np.uint64),
        'annotation_end': np.array([20, 60, 90], dtype=np.uint64),
        'label': np.array([0, 1, 2], dtype=np.int64),
    }
    fake_dataset_info = mock.MagicMock(
        features={
            'audio': mock.MagicMock(sample_rate=10),
            'label': mock.MagicMock(names=('dowwoo', 'daejun', 'pilwoo')),
        }
    )
    original_dataset = tf.data.Dataset.from_tensors(original_example)
    annotated_dataset = pipeline.DenselyAnnotateWindows(
        overlap_threshold_sec=1
    )(original_dataset, fake_dataset_info)
    annotated_dataset = next(annotated_dataset.as_numpy_iterator())

    expected_dataset = {
        'segment_start': np.array(10, dtype=np.uint64),
        'segment_end': np.array(50, dtype=np.uint64),
        # The annotations for labels 0 and 1 are longer than 1s, so are kept.
        # The annotation metadata for label 2 is all zeros.
        'annotation_start': np.array([10, 30, 0], dtype=np.uint64),
        'annotation_end': np.array([20, 60, 0], dtype=np.uint64),
        'intersection_size': np.array([10, 20, 0], dtype=np.uint64),
        'annotation_length': np.array([10, 30, 0], dtype=np.uint64),
        'label': np.array([0, 1], dtype=np.int64),
    }

    for key, expected_value in expected_dataset.items():
      print(key, expected_value, annotated_dataset[key])
      np.testing.assert_equal(expected_value, annotated_dataset[key])

  def test_repeat_padding(self):
    # Set some example values
    sample_rate_hz = self._builder.info.features['audio'].sample_rate
    audio_length_s = 2
    audio_length_samples = sample_rate_hz * audio_length_s
    window_size_s = 5

    # Create a random audio tensor
    example = {
        'audio': tf.random.uniform(
            [audio_length_samples],
            dtype=tf.float32,
        ),
    }

    # Apply RepeatPadding to the audio
    repeat_padding_op = pipeline.RepeatPadding(
        pad_size=window_size_s, sample_rate=sample_rate_hz
    )
    repeat_pad_example = repeat_padding_op(example, self._builder.info)

    # Apply boring old zero padding
    zero_pad_op = pipeline.Pad(
        pad_size=window_size_s, sample_rate=sample_rate_hz
    )
    zero_pad_example = zero_pad_op(example, self._builder.info)

    # Check repeat pad has the right output length
    self.assertEqual(
        tf.shape(repeat_pad_example['audio'])[-1],
        window_size_s * sample_rate_hz,
    )

    # Check that both padding operations result in the same output length
    self.assertEqual(
        tf.shape(repeat_pad_example['audio'])[-1],
        tf.shape(zero_pad_example['audio'])[-1],
    )

  def test_AddTensorOp(self):
    """Test for combined datasets."""
    # Define some sample datasets
    ds1 = tf.data.Dataset.from_tensor_slices({
        'tensor_A': tf.constant([1, 2, 3], dtype=tf.int32),
        'tensor_B': tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
    }).batch(3)
    ds2 = tf.data.Dataset.from_tensor_slices({
        'tensor_A': tf.constant([1, 2, 3], dtype=tf.int32),
        # add new tensor_C
        'tensor_C': tf.constant([1, 2, 3], dtype=tf.int32),
    }).batch(3)
    ds3 = tf.data.Dataset.from_tensor_slices({
        'tensor_A': tf.constant([1, 2, 3], dtype=tf.int32),
        # tensor_B is different shape
        'tensor_B': tf.constant(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=tf.float32
        ),
    }).batch(3)
    ds4 = tf.data.Dataset.from_tensor_slices({
        'tensor_A': tf.constant([1, 2, 3], dtype=tf.int32),
        # tensor_B is different dtype
        'tensor_B': tf.constant([1, 2, 3], dtype=tf.int64),
    }).batch(3)

    # Mock dataset_info. dataset_info is not currently used anywhere.
    mock_dataset_info = mock.Mock()

    # Ensure the unified tensor dict info contains details about all tensors
    merge_op = pipeline.AddTensorOp.from_datasets([ds1, ds2])
    unified_ds_dict = merge_op.unified_shape_info
    assert 'tensor_A' in unified_ds_dict
    assert 'tensor_B' in unified_ds_dict
    assert 'tensor_C' in unified_ds_dict

    # Unify datasets
    unified_ds1 = pipeline.Pipeline([merge_op])(ds1, mock_dataset_info)
    unified_ds2 = pipeline.Pipeline([merge_op])(ds2, mock_dataset_info)

    # Extract one item from unified datasets to check tensors
    example_ds1 = next(iter(unified_ds1))
    example_ds2 = next(iter(unified_ds2))

    # Check tensors in unified_ds1
    assert 'tensor_A' in example_ds1
    assert 'tensor_B' in example_ds1
    assert 'tensor_C' in example_ds1
    assert example_ds1['tensor_A'].dtype == tf.int32
    assert example_ds1['tensor_B'].dtype == tf.float32
    assert example_ds1['tensor_C'].dtype == tf.int32

    # Check tensors in unified_ds2
    assert 'tensor_A' in example_ds2
    assert 'tensor_B' in example_ds2
    assert 'tensor_C' in example_ds2
    assert example_ds2['tensor_A'].dtype == tf.int32
    assert example_ds2['tensor_B'].dtype == tf.float32
    assert example_ds2['tensor_C'].dtype == tf.int32

    # Test error is raised with the faulty ds3 and ds4 datasets
    with self.assertRaises(ValueError):
      pipeline.AddTensorOp.from_datasets([ds1, ds3])
    with self.assertRaises(ValueError):
      pipeline.AddTensorOp.from_datasets([ds1, ds4])

  def test_RemoveUnwantedFeatures(self):
    """Test for removing unwanted keys from the dataset features."""
    # Define a mock dataset and unwanted keys
    features = {
        'tensor_A': tf.constant([1, 2, 3], dtype=tf.int32),
        'tensor_B': tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),
        'tensor_C': tf.constant(['a', 'b', 'c', 'd'], dtype=tf.string),
    }
    unwanted_keys = ['tensor_A', 'tensor_C']

    # Mock dataset_info. dataset_info is not currently used anywhere.
    mock_dataset_info = mock.Mock()

    # Apply the op
    op = pipeline.RemoveUnwantedFeatures(unwanted_keys=unwanted_keys)
    modified_features = op(features, mock_dataset_info)

    # Check keys removed
    assert 'tensor_A' not in modified_features
    assert 'tensor_C' not in modified_features

    # Check unchanged key remains
    assert 'tensor_B' in modified_features
    np.testing.assert_array_equal(
        modified_features['tensor_B'].numpy(), features['tensor_B'].numpy()
    )


if __name__ == '__main__':
  absltest.main()
