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

"""Tests for inference library."""

import os
import tempfile

import apache_beam as beam
from apache_beam.testing import test_pipeline
from chirp import path_utils
from chirp.inference import embed_lib
from chirp.inference import models
from chirp.taxonomy import namespace_db
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
    )
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav'
    )

    source_info = embed_lib.SourceInfo(test_wav_path, 0, 1)
    feature_description = embed_lib.get_feature_description(
        logit_names=['label']
    )

    example = embed_fn.process(source_info, crop_s=10.0)[0]
    serialized = example.SerializeToString()
    got_example = tf.io.parse_single_example(serialized, feature_description)
    self.assertIsNotNone(got_example)
    # got_example = tf.io.parse_single_example(serialized, feature_description)
    self.assertEqual(got_example[embed_lib.FILE_NAME], 'clap.wav')
    if make_embeddings and write_embeddings:
      embedding = tf.io.parse_tensor(got_example['embedding'], tf.float32)
      self.assertSequenceEqual(embedding.shape, got_example['embedding_shape'])
    else:
      self.assertEqual(got_example['embedding'], '')

    if make_logits and write_logits:
      logits = tf.io.parse_tensor(got_example['label'], tf.float32)
      self.assertSequenceEqual(logits.shape, got_example['label_shape'])
    else:
      self.assertEqual(got_example['label'], '')

    if make_separated_audio and write_separated_audio:
      separated_audio = tf.io.parse_tensor(
          got_example['separated_audio'], tf.float32
      )
      self.assertSequenceEqual(
          separated_audio.shape, got_example['separated_audio_shape']
      )
    else:
      self.assertEqual(got_example['separated_audio'], '')

    if write_raw_audio:
      raw_audio = tf.io.parse_tensor(got_example['raw_audio'], tf.float32)
      self.assertSequenceEqual(raw_audio.shape, got_example['raw_audio_shape'])
    else:
      self.assertEqual(got_example['raw_audio'], '')

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
        sample_rate=32000,
        make_embeddings=True,
        make_logits=True,
        make_separated_audio=False,
        target_class_list=target_class_list,
    )
    sep_embed = models.SeparateEmbedModel(22050, separator, embeddor)
    audio = np.zeros(5 * 22050, np.float32)

    sep_outputs = separator.embed(audio)
    outputs = sep_embed.embed(audio)
    self.assertSequenceEqual(
        sep_outputs.separated_audio.shape, outputs.separated_audio.shape
    )
    # The PlaceholderModel produces one embedding per second, and we have
    # five seconds of audio, with two separated channels.
    # Note that this checks that the sample-rate conversion between the
    # separation model and embedding model has worked correctly.
    self.assertSequenceEqual(
        outputs.embeddings.shape, [5, 2, embeddor.embedding_size]
    )
    # The Sep+Embed model takes the max logits over the channel dimension.
    self.assertSequenceEqual(
        outputs.logits['label'].shape, [5, target_class_list.size]
    )

  def test_beam_pipeline(self):
    """Check that we can write embeddings to TFRecord file."""
    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav'
    )
    source_infos = [embed_lib.SourceInfo(test_wav_path.as_posix(), 0, 0)]
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
    counter = counter = metrics.query(
        beam.metrics.MetricsFilter().with_name('examples_processed')
    )['counters']
    self.assertEqual(counter[0].result, 1)

    print(metrics)


if __name__ == '__main__':
  absltest.main()
