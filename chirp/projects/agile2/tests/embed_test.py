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

"""Tests for embedding audio."""

import shutil
import tempfile

from chirp.projects.agile2 import embed
from chirp.projects.agile2.tests import test_utils
from chirp.projects.hoplite import db_loader
from ml_collections import config_dict

from absl.testing import absltest


class EmbedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_embed_worker(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    test_utils.make_wav_files(self.tempdir, classes, filenames, file_len_s=6.0)

    audio_globs = {'test': (self.tempdir, '*/*.wav')}
    embed_config = embed.EmbedConfig(
        audio_globs=audio_globs,
        min_audio_len_s=0.0,
        target_sample_rate_hz=16000,
    )

    in_mem_db_config = config_dict.ConfigDict()
    in_mem_db_config.embedding_dim = 32
    in_mem_db_config.max_size = 100
    in_mem_db_config.degree_bound = 64
    db_config = db_loader.DBConfig(
        db_key='in_mem',
        db_config=in_mem_db_config,
    )

    placeholder_model_config = config_dict.ConfigDict()
    placeholder_model_config.embedding_size = 32
    placeholder_model_config.sample_rate = 16000
    model_config = embed.ModelConfig(
        model_key='placeholder_model',
        model_config=placeholder_model_config,
    )

    embed_worker = embed.EmbedWorker(
        embed_config=embed_config,
        model_config=model_config,
        db_config=db_config,
    )
    embed_worker.process_all()
    # The hop size is 1.0s and each file is 6.0s, so we should get 6 embeddings
    # per file. There are six files, so we should get 36 embeddings.
    self.assertEqual(embed_worker.db.count_embeddings(), 36)


if __name__ == '__main__':
  absltest.main()
