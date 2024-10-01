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
from chirp.inference import tf_examples
from chirp.inference.classify import classify
from chirp.inference.classify import data_lib
from chirp.inference.search import bootstrap
from chirp.inference.search import display
from chirp.inference.search import search
from chirp.models import metrics
from chirp.projects.zoo import models
from chirp.projects.zoo import taxonomy_model_tf
from chirp.projects.zoo import zoo_interface
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
  return zoo_interface.LogitsOutputHead(
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
      restored_model = zoo_interface.LogitsOutputHead.from_config_file(
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
      outputs = zoo_interface.InferenceOutputs(
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
      outputs = zoo_interface.InferenceOutputs(
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


if __name__ == '__main__':
  absltest.main()
