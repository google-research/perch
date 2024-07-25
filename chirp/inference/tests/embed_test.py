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

"""Tests for mass-embedding functionality."""

import os
import tempfile

import apache_beam as beam
from apache_beam.testing import test_pipeline
from chirp import audio_utils
from chirp import config_utils
from chirp import path_utils
from chirp.inference import colab_utils
from chirp.inference import embed_lib
from chirp.inference import interface
from chirp.inference import models
from chirp.inference import tf_examples
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib
from chirp.inference.search import bootstrap
from chirp.inference.search import display
from chirp.inference.search import search
from chirp.models import metrics
from chirp.taxonomy import namespace
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tf_keras

from absl.testing import absltest
from absl.testing import parameterized


def _make_output_head_model(model_path: str, embedding_dim: int = 1280):
  classes = ('speech', 'birdsong', 'unknown')
  model = tf.keras.Sequential([
      tf.keras.Input(shape=[embedding_dim]),
      tf.keras.layers.Dense(len(classes)),
  ])
  class_list = namespace.ClassList('custom', classes)
  return interface.LogitsOutputHead(
      model_path, 'other_label', model, class_list
  )


class EmbedTest(parameterized.TestCase):

  def test_imports(self):
    # Test that imports work in external github environment.
    # This explicitly tests that libraries commonly used in Colab workflows
    # can be imported when Perch is installed without Jax training dependencies.
    self.assertIsNotNone(audio_utils)
    self.assertIsNotNone(bootstrap)
    self.assertIsNotNone(classify)
    self.assertIsNotNone(colab_utils)
    self.assertIsNotNone(config_utils)
    self.assertIsNotNone(data_lib)
    self.assertIsNotNone(display)
    self.assertIsNotNone(metrics)
    self.assertIsNotNone(search)

  @parameterized.parameters(
      # Test each output type individually.
      {'make_embeddings': True},
      {'make_embeddings': True, 'write_embeddings': True},
      {'make_logits': True},
      {'make_logits': True, 'write_logits': True},
      {'make_separated_audio': True},
      {'make_separated_audio': True, 'write_separated_audio': True},
      {'make_frontend': True},
      {'make_frontend': True, 'write_frontend': True},
      {'write_raw_audio': True},
      # Check float16 handling.
      {'make_embeddings': True, 'tensor_dtype': 'float16'},
      {
          'make_embeddings': True,
          'write_embeddings': True,
          'tensor_dtype': 'float16',
      },
      # Check with all active.
      {
          'make_embeddings': True,
          'make_logits': True,
          'make_separated_audio': True,
          'make_frontend': True,
          'write_embeddings': True,
          'write_logits': True,
          'write_separated_audio': True,
          'write_frontend': True,
          'write_raw_audio': True,
      },
  )
  def test_embed_fn(
      self,
      make_embeddings=False,
      make_logits=False,
      make_separated_audio=False,
      make_frontend=False,
      write_embeddings=False,
      write_logits=False,
      write_raw_audio=False,
      write_separated_audio=False,
      write_frontend=False,
      tensor_dtype='float32',
  ):
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': make_embeddings,
        'make_logits': make_logits,
        'make_separated_audio': make_separated_audio,
        'make_frontend': make_frontend,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=write_embeddings,
        write_logits=write_logits,
        write_separated_audio=write_separated_audio,
        write_raw_audio=write_raw_audio,
        write_frontend=write_frontend,
        model_key='placeholder_model',
        model_config=model_kwargs,
        file_id_depth=0,
        tensor_dtype=tensor_dtype,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )

    source_info = embed_lib.SourceInfo(test_wav_path, 0, 10)
    example = embed_fn.process(source_info, crop_s=10.0)[0]
    serialized = example.SerializeToString()

    parser = tf_examples.get_example_parser(
        logit_names=['label', 'other_label'],
        tensor_dtype=tensor_dtype,
    )
    got_example = parser(serialized)
    self.assertIsNotNone(got_example)
    self.assertEqual(got_example[tf_examples.FILE_NAME], 'clap.wav')
    if make_embeddings and write_embeddings:
      embedding = got_example[tf_examples.EMBEDDING]
      self.assertSequenceEqual(
          embedding.shape, got_example[tf_examples.EMBEDDING_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.EMBEDDING].shape, (0,))

    if make_logits and write_logits:
      self.assertSequenceEqual(
          got_example['label'].shape, got_example['label_shape']
      )
      self.assertSequenceEqual(
          got_example['other_label'].shape, got_example['other_label_shape']
      )
    else:
      self.assertEqual(got_example['label'].shape, (0,))

    if make_separated_audio and write_separated_audio:
      separated_audio = got_example[tf_examples.SEPARATED_AUDIO]
      self.assertSequenceEqual(
          separated_audio.shape, got_example[tf_examples.SEPARATED_AUDIO_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.SEPARATED_AUDIO].shape, (0,))

    if make_frontend and write_frontend:
      frontend = got_example[tf_examples.FRONTEND]
      self.assertSequenceEqual(
          frontend.shape, got_example[tf_examples.FRONTEND_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.FRONTEND].shape, (0,))

    if write_raw_audio:
      raw_audio = got_example[tf_examples.RAW_AUDIO]
      self.assertSequenceEqual(
          raw_audio.shape, got_example[tf_examples.RAW_AUDIO_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.RAW_AUDIO].shape, (0,))

  @parameterized.parameters(
      {'config_filename': 'embedding_config_v0'},
      # Includes frontend handling options.
      {'config_filename': 'embedding_config_v1'},
  )
  def test_embed_fn_from_config(self, config_filename):
    # Test that we can load a model from a golden config and compute embeddings.
    test_config_path = os.fspath(
        path_utils.get_absolute_path(
            f'inference/tests/testdata/{config_filename}.json'
        )
    )
    embed_config = embed_lib.load_embedding_config(test_config_path, '')
    embed_fn = embed_lib.EmbedFn(**embed_config)
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )
    source_info = embed_lib.SourceInfo(test_wav_path, 0, 10)
    example = embed_fn.process(source_info, crop_s=10.0)[0]
    serialized = example.SerializeToString()

    parser = tf_examples.get_example_parser(
        logit_names=['label', 'other_label'],
        tensor_dtype=embed_config.tensor_dtype,
    )
    got_example = parser(serialized)
    self.assertIsNotNone(got_example)

  def test_embed_fn_source_variations(self):
    """Test processing with variations of SourceInfo."""
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=True,
        write_logits=False,
        write_separated_audio=False,
        write_raw_audio=False,
        write_frontend=False,
        model_key='placeholder_model',
        model_config=model_kwargs,
        min_audio_s=2.0,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)
    parser = tf_examples.get_example_parser()

    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )

    # Check that a SourceInfo with window_size_s <= 0 embeds the entire file.
    source_info = embed_lib.SourceInfo(test_wav_path, 0, -1)
    example = embed_fn.process(source_info)[0]
    example = parser(example.SerializeToString())
    self.assertEqual(example['embedding'].shape[0], 21)

    # Check that a SourceInfo with window_size_s > 0 embeds part of the file.
    source_info = embed_lib.SourceInfo(test_wav_path, 0, 5.0)
    example = embed_fn.process(source_info)[0]
    example = parser(example.SerializeToString())
    self.assertEqual(example['embedding'].shape[0], 5)

    # Check that the second part of the file has the correct length.
    source_info = embed_lib.SourceInfo(test_wav_path, 1, 5.0)
    example = embed_fn.process(source_info)[0]
    example = parser(example.SerializeToString())
    self.assertEqual(example['embedding'].shape[0], 5)

    # Check that the end of a file is properly handled.
    # In this case, the window_size_s is 6.0, and we have a 21 audio file.
    # So shard number 3 should be the 3s remainder.
    source_info = embed_lib.SourceInfo(test_wav_path, 3, 6.0)
    example = embed_fn.process(source_info)[0]
    example = parser(example.SerializeToString())
    self.assertEqual(example['embedding'].shape[0], 3)

    # Check that a too-short remainder returns None.
    # In this case, the window_size_s is 5.0, and we have a 21s audio file.
    # So shard number 4 should be the 1s remainder, but the min_audio_s is set
    # to 2s, so we should drop the example.
    source_info = embed_lib.SourceInfo(test_wav_path, 4, 5.0)
    self.assertIsNone(embed_fn.process(source_info))

  def test_keyed_write_logits(self):
    """Test that EmbedFn writes only the desired logits if specified."""
    write_logits = ('other_label',)
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': ('label', 'other_label'),
        'make_separated_audio': False,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=True,
        write_logits=write_logits,
        write_separated_audio=False,
        write_raw_audio=False,
        write_frontend=False,
        model_key='placeholder_model',
        model_config=model_kwargs,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )

    source_info = embed_lib.SourceInfo(test_wav_path, 0, 10)
    example = embed_fn.process(source_info, crop_s=10.0)[0]
    serialized = example.SerializeToString()

    parser = tf_examples.get_example_parser(
        logit_names=['label', 'other_label']
    )
    got_example = parser(serialized)
    self.assertIsNotNone(got_example)
    self.assertEqual(got_example[tf_examples.FILE_NAME], 'clap.wav')
    self.assertSequenceEqual(
        got_example['other_label'].shape, got_example['other_label_shape']
    )
    self.assertEqual(got_example['label'].shape, (0,))

  def test_logits_output_head(self):
    base_model = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=True,
        make_logits=False,
        make_separated_audio=True,
    )
    logits_model = _make_output_head_model(
        '/tmp/logits_model', embedding_dim=128
    )
    base_outputs = base_model.embed(np.zeros(5 * 22050))
    updated_outputs = logits_model.add_logits(base_outputs, keep_original=True)
    self.assertSequenceEqual(
        updated_outputs.logits['other_label'].shape,
        (5, 3),
    )
    # Check that we /only/ have the new logits, since make_logits=False
    self.assertNotIn('label', updated_outputs.logits)

    # Save and restore the model.
    with tempfile.TemporaryDirectory() as logits_model_dir:
      logits_model.save_model(logits_model_dir, '')
      restored_model = interface.LogitsOutputHead.from_config_file(
          logits_model_dir
      )
    reupdated_outputs = restored_model.add_logits(
        base_outputs, keep_original=True
    )
    error = np.mean(
        np.abs(
            reupdated_outputs.logits['other_label']
            - updated_outputs.logits['other_label']
        )
    )
    self.assertLess(error, 1e-5)

  def test_embed_short_audio(self):
    """Test that EmbedFn handles audio shorter than the model window_size_s."""
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
        'window_size_s': 5.0,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=True,
        write_logits=False,
        write_separated_audio=False,
        write_raw_audio=False,
        write_frontend=False,
        model_key='placeholder_model',
        model_config=model_kwargs,
        min_audio_s=1.0,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )
    source_info = embed_lib.SourceInfo(test_wav_path, 0, 10)
    # Crop to 3.0s to ensure we can handle short audio examples.
    example = embed_fn.process(source_info, crop_s=3.0)[0]
    serialized = example.SerializeToString()

    parser = tf_examples.get_example_parser(logit_names=['label'])
    got_example = parser(serialized)
    self.assertIsNotNone(got_example)
    embedding = got_example[tf_examples.EMBEDDING]
    self.assertSequenceEqual(
        embedding.shape, got_example[tf_examples.EMBEDDING_SHAPE]
    )

  def test_frame_audio(self):
    """Test that EmbedModel correctly frames a short audio chunk."""
    model_kwargs = {
        'sample_rate': 200,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
        'do_frame_audio': True,
        'window_size_s': 5.0,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=True,
        write_logits=False,
        write_separated_audio=False,
        write_raw_audio=False,
        write_frontend=False,
        model_key='placeholder_model',
        model_config=model_kwargs,
        min_audio_s=1.0,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    framed = embed_fn.embedding_model.frame_audio(np.ones(100), 1.0, 5.0)
    self.assertEqual(framed.shape, (1, 200))

  def test_create_source_infos(self):
    # Just one file, but it's all good.
    globs = [
        path_utils.get_absolute_path(
            'inference/tests/testdata/clap.wav'
        ).as_posix()
    ]
    # Disable sharding by setting shard_len_s <= 0.
    got_infos = embed_lib.create_source_infos(
        globs, shard_len_s=-1, num_shards_per_file=100
    )
    self.assertLen(got_infos, 1)

    # Try automatic sharding by setting num_shards_per_file < 0.
    got_infos = embed_lib.create_source_infos(
        globs, shard_len_s=10, num_shards_per_file=-1
    )
    # The test file is ~21s long, so we should get three shards.
    self.assertLen(got_infos, 3)

    # Use a fixed number of shards per file.
    got_infos = embed_lib.create_source_infos(
        globs, shard_len_s=10, num_shards_per_file=10
    )
    self.assertLen(got_infos, 10)

  def test_tfrecord_multiwriter(self):
    output_dir = epath.Path(tempfile.TemporaryDirectory().name)
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_examples = []
    for idx in range(20):
      outputs = interface.InferenceOutputs(
          embeddings=np.zeros([10, 2, 8], dtype=np.float32), batched=False
      )
      fake_examples.append(
          tf_examples.model_outputs_to_tf_example(
              model_outputs=outputs,
              file_id=f'fake_audio_{idx:02d}',
              audio=np.zeros([100]),
              timestamp_offset_s=0.0,
              write_embeddings=True,
              write_logits=False,
              write_separated_audio=False,
              write_raw_audio=False,
              write_frontend=False,
          )
      )
    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir.as_posix()
    ) as writer:
      for ex in fake_examples:
        serialized = ex.SerializeToString()
        writer.write(serialized)

    fns = [fn for fn in output_dir.glob('embeddings-*')]
    ds = tf.data.TFRecordDataset(fns)
    parser = tf_examples.get_example_parser()
    ds = ds.map(parser)

    got_examples = [ex for ex in ds.as_numpy_iterator()]
    self.assertEqual(len(got_examples), len(fake_examples))
    want_ids = [f'fake_audio_{idx:02d}' for idx in range(20)]
    got_ids = sorted([ex['filename'].decode('utf-8') for ex in got_examples])
    self.assertSequenceEqual(want_ids, got_ids)

  def test_get_existing_source_ids(self):
    output_dir = epath.Path(tempfile.TemporaryDirectory().name)
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_examples = []
    for idx in range(20):
      outputs = interface.InferenceOutputs(
          embeddings=np.zeros([10, 2, 8], dtype=np.float32), batched=False
      )
      fake_examples.append(
          tf_examples.model_outputs_to_tf_example(
              model_outputs=outputs,
              file_id=f'fake_audio_{idx:02d}',
              audio=np.zeros([100]),
              timestamp_offset_s=float(idx),
              write_embeddings=False,
              write_logits=False,
              write_separated_audio=False,
              write_raw_audio=False,
              write_frontend=False,
          )
      )
    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir.as_posix()
    ) as writer:
      for ex in fake_examples:
        serialized = ex.SerializeToString()
        writer.write(serialized)

    actual_ids = embed_lib.get_existing_source_ids(output_dir, 'embeddings-*')

    expected_ids = set(
        [
            embed_lib.SourceId(f'fake_audio_{idx:02d}', float(idx))
            for idx in range(20)
        ]
    )
    self.assertSetEqual(expected_ids, actual_ids)

  def test_get_new_source_ids(self):
    all_infos = [
        embed_lib.SourceInfo(f'fake_audio_{idx:02d}', idx, shard_len_s=1.0)
        for idx in range(20)
    ]
    existing_ids = set(
        [
            embed_lib.SourceId(f'fake_audio_{idx:02d}', float(idx))
            for idx in range(10)
        ]
    )

    actual_infos = embed_lib.get_new_source_infos(all_infos, existing_ids, 0)
    expected_infos = all_infos[10:]
    self.assertSequenceEqual(expected_infos, actual_infos)

  @parameterized.product(
      config_name=(
          'raw_soundscapes',
          'separate_soundscapes',
          'birdnet_soundscapes',
      ),
  )
  def test_load_configs(self, config_name):
    config = embed_lib.get_config(config_name)
    self.assertIsNotNone(config)

  def test_handcrafted_features(self):
    model = models.HandcraftedFeaturesModel.beans_baseline()

    audio = np.zeros([5 * 32000], dtype=np.float32)
    outputs = model.embed(audio)
    # Five frames because we have 5s of audio with window 1.0 and hope 1.0.
    # Beans aggrregation with mfccs creates 20 MFCC channels, and then computes
    # four summary statistics for each, giving a total of 80 output channels.
    self.assertSequenceEqual([5, 1, 80], outputs.embeddings.shape)

  def test_sep_embed_wrapper(self):
    """Check that the joint-model wrapper works as intended."""
    separator = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=False,
        make_logits=False,
        make_separated_audio=True,
    )

    embeddor = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=True,
        make_logits=True,
        make_separated_audio=False,
    )
    fake_config = config_dict.ConfigDict()
    sep_embed = models.SeparateEmbedModel(
        sample_rate=22050,
        taxonomy_model_tf_config=fake_config,
        separator_model_tf_config=fake_config,
        separation_model=separator,
        embedding_model=embeddor,
    )
    audio = np.zeros(5 * 22050, np.float32)

    outputs = sep_embed.embed(audio)
    # The PlaceholderModel produces one embedding per second, and we have
    # five seconds of audio, with two separated channels, plus the channel
    # for the raw audio.
    # Note that this checks that the sample-rate conversion between the
    # separation model and embedding model has worked correctly.
    self.assertSequenceEqual(
        outputs.embeddings.shape, [5, 3, embeddor.embedding_size]
    )
    # The Sep+Embed model takes the max logits over the channel dimension.
    self.assertSequenceEqual(
        outputs.logits['label'].shape, [5, len(embeddor.class_list.classes)]
    )

  def test_pooled_embeddings(self):
    outputs = interface.InferenceOutputs(
        embeddings=np.zeros([10, 2, 8]), batched=False
    )
    batched_outputs = interface.InferenceOutputs(
        embeddings=np.zeros([3, 10, 2, 8]), batched=True
    )

    # Check that no-op is no-op.
    non_pooled = outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(non_pooled.shape, outputs.embeddings.shape)
    batched_non_pooled = batched_outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(
        batched_non_pooled.shape, batched_outputs.embeddings.shape
    )

    for pooling_method in interface.POOLING_METHODS:
      if pooling_method == 'squeeze':
        # The 'squeeze' pooling method throws an exception if axis size is > 1.
        with self.assertRaises(ValueError):
          outputs.pooled_embeddings(pooling_method, '')
        continue
      elif pooling_method == 'flatten':
        # Concatenates over the target axis.
        time_pooled = outputs.pooled_embeddings(pooling_method, '')
        self.assertSequenceEqual(time_pooled.shape, [2, 80])
        continue

      time_pooled = outputs.pooled_embeddings(pooling_method, '')
      self.assertSequenceEqual(time_pooled.shape, [2, 8])
      batched_time_pooled = batched_outputs.pooled_embeddings(
          pooling_method, ''
      )
      self.assertSequenceEqual(batched_time_pooled.shape, [3, 2, 8])

      channel_pooled = outputs.pooled_embeddings('', pooling_method)
      self.assertSequenceEqual(channel_pooled.shape, [10, 8])
      batched_channel_pooled = batched_outputs.pooled_embeddings(
          '', pooling_method
      )
      self.assertSequenceEqual(batched_channel_pooled.shape, [3, 10, 8])

      both_pooled = outputs.pooled_embeddings(pooling_method, pooling_method)
      self.assertSequenceEqual(both_pooled.shape, [8])
      batched_both_pooled = batched_outputs.pooled_embeddings(
          pooling_method, pooling_method
      )
      self.assertSequenceEqual(batched_both_pooled.shape, [3, 8])

  def test_beam_pipeline(self):
    """Check that we can write embeddings to TFRecord file."""
    test_wav_path = os.fspath(
        path_utils.get_absolute_path('inference/tests/testdata/clap.wav')
    )
    source_infos = [embed_lib.SourceInfo(test_wav_path, 0, 10)]
    base_pipeline = test_pipeline.TestPipeline()
    tempdir = tempfile.gettempdir()
    output_dir = os.path.join(tempdir, 'testBeamStuff_output')

    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=False,
        write_logits=False,
        write_separated_audio=False,
        write_raw_audio=False,
        write_frontend=False,
        model_key='placeholder_model',
        model_config=model_kwargs,
        file_id_depth=0,
    )

    metrics = embed_lib.build_run_pipeline(
        base_pipeline, output_dir, source_infos, embed_fn
    )
    counter = metrics.query(
        beam.metrics.MetricsFilter().with_name('examples_processed')
    )['counters']
    self.assertEqual(counter[0].result, 1)

    print(metrics)

  @parameterized.product(
      model_return_type=('tuple', 'dict'),
      batchable=(True, False),
  )
  def test_taxonomy_model_tf(self, model_return_type, batchable):
    class FakeModelFn:
      output_depths = {'label': 3, 'embedding': 256}

      def infer_tf(self, audio_array):
        outputs = {
            k: np.zeros([audio_array.shape[0], d], dtype=np.float32)
            for k, d in self.output_depths.items()
        }
        if model_return_type == 'tuple':
          # Published Perch models v1 through v4 returned a tuple, not a dict.
          return outputs['label'], outputs['embedding']
        return outputs

    class_list = {
        'label': namespace.ClassList('fake', ['alpha', 'beta', 'delta'])
    }
    wrapped_model = models.TaxonomyModelTF(
        sample_rate=32000,
        model_path='/dev/null',
        window_size_s=5.0,
        hop_size_s=5.0,
        model=FakeModelFn(),
        class_list=class_list,
        batchable=batchable,
    )

    # Check that a single frame of audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([5 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [1, 3])

    # Check that multi-frame audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([20 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [4, 3])

    # Check that a batch of single frame of audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([10, 5 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [10, 1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [10, 1, 3])

    # Check that a batch of multi-frame audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([2, 20 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [2, 4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [2, 4, 3])

  def test_whale_model(self):
    # prereq
    class FakeModel(tf_keras.Model):
      """Fake implementation of the humpback_whale SavedModel API.

      The use of `tf_keras` as opposed to `tf.keras` is intentional; the models
      this fakes were exported using "the pure-TensorFlow implementation of
      Keras."
      """

      def __init__(self):
        super().__init__()
        self._sample_rate = 10000
        self._classes = ['Mn']
        self._embedder = tf_keras.layers.Dense(32)
        self._classifier = tf_keras.layers.Dense(len(self._classes))

      def call(self, spectrograms, training=False):
        logits = self.logits(spectrograms)
        return tf.nn.sigmoid(logits)

      @tf.function(
          input_signature=[tf.TensorSpec([None, None, 1], tf.dtypes.float32)]
      )
      def front_end(self, waveform):
        return tf.math.abs(
            tf.signal.stft(
                tf.squeeze(waveform, -1),
                frame_length=1024,
                frame_step=300,
                fft_length=128,
            )[..., 1:]
        )

      @tf.function(
          input_signature=[tf.TensorSpec([None, 128, 64], tf.dtypes.float32)]
      )
      def features(self, spectrogram):
        return self._embedder(tf.math.reduce_mean(spectrogram, axis=-2))

      @tf.function(
          input_signature=[tf.TensorSpec([None, 128, 64], tf.dtypes.float32)]
      )
      def logits(self, spectrogram):
        features = self.features(spectrogram)
        return self._classifier(features)

      @tf.function(
          input_signature=[
              tf.TensorSpec([None, None, 1], tf.dtypes.float32),
              tf.TensorSpec([], tf.dtypes.int64),
          ]
      )
      def score(self, waveform, context_step_samples):
        spectrogram = self.front_end(waveform)
        windows = tf.signal.frame(
            spectrogram, frame_length=128, frame_step=128, axis=1
        )
        shape = tf.shape(windows)
        batch_size = shape[0]
        num_windows = shape[1]
        frame_length = shape[2]
        tf.debugging.assert_equal(frame_length, 128)
        channels_len = shape[3]
        logits = self.logits(
            tf.reshape(
                windows, (batch_size * num_windows, frame_length, channels_len)
            )
        )
        return {'score': tf.nn.sigmoid(logits)}

      @tf.function(input_signature=[])
      def metadata(self):
        return {
            'input_sample_rate': tf.constant(
                self._sample_rate, tf.dtypes.int64
            ),
            'context_width_samples': tf.constant(39124, tf.dtypes.int64),
            'class_names': tf.constant(self._classes),
        }

    # setup
    fake_model = FakeModel()
    batch_size = 2
    duration_seconds = 10
    sample_rate = fake_model.metadata()['input_sample_rate']
    waveform = np.random.randn(
        batch_size,
        sample_rate * duration_seconds,
    )
    expected_frames = int(10 / 3.9124) + 1
    # Call the model to avoid "forward pass of the model is not defined" on
    # save.
    spectrograms = fake_model.front_end(waveform[:, :, np.newaxis])
    fake_model(spectrograms[:, :128, :])
    model_path = os.path.join(tempfile.gettempdir(), 'whale_model')
    fake_model.save(
        model_path,
        signatures={
            'score': fake_model.score,
            'metadata': fake_model.metadata,
            'serving_default': fake_model.score,
            'front_end': fake_model.front_end,
            'features': fake_model.features,
            'logits': fake_model.logits,
        },
    )

    with self.subTest('from_url'):
      # invoke
      model = models.GoogleWhaleModel.load_humpback_model(model_path)
      outputs = model.batch_embed(waveform)

      # verify
      self.assertTrue(outputs.batched)
      self.assertSequenceEqual(
          outputs.embeddings.shape, [batch_size, expected_frames, 1, 32]
      )
      self.assertSequenceEqual(
          outputs.logits['humpback'].shape, [batch_size, expected_frames, 1]
      )

    with self.subTest('from_config'):
      # invoke
      config = config_dict.ConfigDict()
      config.model_url = model_path
      config.sample_rate = float(sample_rate)
      config.window_size_s = 3.9124
      config.peak_norm = 0.02
      class_list = namespace.ClassList('humpback', ['mooooooooohhhhhaaaaaaa'])
      config.class_list = class_list
      model = models.GoogleWhaleModel.from_config(config)
      # Let's check the regular embed this time.
      outputs = model.embed(waveform[0])

      # verify
      self.assertFalse(outputs.batched)
      self.assertSequenceEqual(
          outputs.embeddings.shape, [expected_frames, 1, 32]
      )
      self.assertSequenceEqual(
          outputs.logits['humpback'].shape, [expected_frames, 1]
      )


if __name__ == '__main__':
  absltest.main()
