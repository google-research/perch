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

from chirp import path_utils
from chirp.inference import embed_lib
import tensorflow as tf

from absl.testing import absltest


class InferenceTest(absltest.TestCase):

  def test_embed_fn(self):
    model_kwargs = {
        'sample_rate': 16000,
        'window_size_s': 5,
        'embedding_size': 128,
    }
    embed_fn = embed_lib.EmbedFn(
        hop_size_s=2.5,
        write_embeddings=True,
        write_logits=True,
        write_separated_audio=True,
        model_key='dummy_model',
        model_config=model_kwargs)
    embed_fn.setup()
    self.assertIsNotNone(embed_fn.embedding_model)

    test_wav_path = path_utils.get_absolute_epath(
        'tests/testdata/tfds_builder_wav_directory_test/clap.wav')

    source_info = embed_lib.SourceInfo(test_wav_path, 0, 1)
    feature_description = embed_lib.get_feature_description()
    serialized = None
    count = 0
    for serialized in embed_fn.process(source_info, crop_s=10):
      self.assertIsNotNone(serialized)
      got_example = tf.io.parse_single_example(serialized, feature_description)
      self.assertEqual(got_example[embed_lib.FILE_NAME], 'clap.wav')
      embedding = tf.io.parse_tensor(got_example['embedding'], tf.float32)
      self.assertSequenceEqual(embedding.shape, [128])
      count += 1
    self.assertEqual(count, 5)


if __name__ == '__main__':
  absltest.main()
