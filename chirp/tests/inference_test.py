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

"""Tests for inference library."""

import os
import tempfile

import apache_beam as beam
from apache_beam.testing import test_pipeline
from chirp import path_utils
from chirp.inference import embed_lib
from chirp.inference import interface
from chirp.inference import models
from chirp.inference import tf_examples
from chirp.models import frontend
from chirp.taxonomy import namespace_db
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class InferenceTest(parameterized.TestCase):

  @parameterized.product(
      make_embeddings=(True, False),
      make_logits=(True, False),
      make_separated_audio=(True, False),
      write_embeddings=(True, False),
      write_logits=(True, False),
      write_separated_audio=(True, False),
      write_raw_audio=(True, False),
  )
  def test_embed_fn(
      self,
      make_embeddings,
      make_logits,
      make_separated_audio,
      write_embeddings,
      write_logits,
      write_raw_audio,
      write_separated_audio,
  ):
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': make_embeddings,
        'make_logits': make_logits,
        'make_separated_audio': make_separated_audio,
    }
    embed_fn = embed_lib.EmbedFn(
        write_embeddings=write_embeddings,
        write_logits=write_logits,
        write_separated_audio=write_separated_audio,
        write_raw_audio=write_raw_audio,
        model_key='placeholder_model',
        model_config=model_kwargs,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav'
    )

    source_info = embed_lib.SourceInfo(test_wav_path.as_posix(), 0, 10)
    example = embed_fn.process(source_info, crop_s=10.0)[0]
    serialized = example.SerializeToString()

    parser = tf_examples.get_example_parser(logit_names=['label'])
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
    else:
      self.assertEqual(got_example['label'].shape, (0,))

    if make_separated_audio and write_separated_audio:
      separated_audio = got_example[tf_examples.SEPARATED_AUDIO]
      self.assertSequenceEqual(
          separated_audio.shape, got_example[tf_examples.SEPARATED_AUDIO_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.SEPARATED_AUDIO].shape, (0,))

    if write_raw_audio:
      raw_audio = got_example[tf_examples.RAW_AUDIO]
      self.assertSequenceEqual(
          raw_audio.shape, got_example[tf_examples.RAW_AUDIO_SHAPE]
      )
    else:
      self.assertEqual(got_example[tf_examples.RAW_AUDIO].shape, (0,))

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
        model_key='placeholder_model',
        model_config=model_kwargs,
        min_audio_s=1.0,
        file_id_depth=0,
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav'
    )
    source_info = embed_lib.SourceInfo(test_wav_path.as_posix(), 0, 10)
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

  @parameterized.product(
      config_name=(
          'raw_soundscapes',
          'separate_soundscapes',
          'birdnet_soundscapes',
          'reef',
      ),
  )
  def test_load_configs(self, config_name):
    config = embed_lib.get_config(config_name)
    self.assertIsNotNone(config)

  def test_handcrafted_features(self):
    sample_rate = 32000
    frame_rate = 100
    mel_config = {
        'sample_rate': sample_rate,
        'features': 160,
        'stride': sample_rate // frame_rate,
        'kernel_size': int(0.08 * sample_rate),
        'freq_range': (60.0, sample_rate / 2.0),
        'scaling_config': frontend.LogScalingConfig(),
    }
    features_config = {
        'compute_mfccs': True,
        'aggregation': 'beans',
    }
    model = models.HandcraftedFeaturesModel(
        sample_rate=sample_rate,
        window_size_s=1.0,
        hop_size_s=1.0,
        melspec_config=mel_config,
        features_config=features_config,
    )

    audio = np.zeros([5 * sample_rate], dtype=np.float32)
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
    db = namespace_db.load_db()
    target_class_list = db.class_lists['high_sierras']

    embeddor = models.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=True,
        make_logits=True,
        make_separated_audio=False,
        target_class_list=target_class_list,
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
        outputs.logits['label'].shape, [5, target_class_list.size]
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
          time_pooled = outputs.pooled_embeddings(pooling_method, '')
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
    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav'
    )
    source_infos = [embed_lib.SourceInfo(test_wav_path.as_posix(), 0, 10)]
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
        model_key='placeholder_model',
        model_config=model_kwargs,
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
