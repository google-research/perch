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

"""Tests for source_info."""

import shutil
import tempfile

from chirp.projects.agile2 import source_info
from chirp.projects.agile2.tests import test_utils

from absl.testing import absltest


class SourceInfoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_audio_sources_iteration(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    test_utils.make_wav_files(self.tempdir, classes, filenames, file_len_s=6.0)
    audio_sources = source_info.AudioSources(
        audio_globs={
            'pos': (self.tempdir, 'pos/*.wav'),
            'neg': (self.tempdir, 'neg/*.wav'),
        }
    )

    with self.subTest('no_sharding'):
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, len(classes) * len(filenames))

    with self.subTest('max_shards_less_than_file_len'):
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
              shard_len_s=2.0, max_shards_per_file=2
          )
      )
      self.assertLen(shard_ids, 12)

    with self.subTest('max_shards_matches_file_len'):
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
              shard_len_s=2.0, max_shards_per_file=3
          )
      )
      self.assertLen(shard_ids, 18)

    with self.subTest('max_shards_larger_than_file_len'):
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
              shard_len_s=2.0, max_shards_per_file=100
          )
      )
      self.assertLen(shard_ids, 18)

    with self.subTest('sharded_no_max_shards'):
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
              shard_len_s=2.0, max_shards_per_file=-1
          )
      )
      self.assertLen(shard_ids, 18)


if __name__ == '__main__':
  absltest.main()
