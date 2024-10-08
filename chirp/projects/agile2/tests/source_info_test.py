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
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='pos',
                base_path=self.tempdir,
                file_glob='pos/*.wav',
            ),
            source_info.AudioSourceConfig(
                dataset_name='neg',
                base_path=self.tempdir,
                file_glob='neg/*.wav',
            ),
        )
    )

    with self.subTest('no_sharding'):
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, len(classes) * len(filenames))

    audio_sources.audio_globs[0].shard_len_s = 2.0
    audio_sources.audio_globs[1].shard_len_s = 2.0
    with self.subTest('max_shards_less_than_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 2
      audio_sources.audio_globs[1].max_shards_per_file = 2
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 12)

    with self.subTest('max_shards_matches_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 3
      audio_sources.audio_globs[1].max_shards_per_file = 3
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 18)

    with self.subTest('max_shards_larger_than_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 100
      audio_sources.audio_globs[1].max_shards_per_file = 100
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
          )
      )
      self.assertLen(shard_ids, 18)

    with self.subTest('sharded_no_max_shards'):
      audio_sources.audio_globs[0].max_shards_per_file = None
      audio_sources.audio_globs[1].max_shards_per_file = None

      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 18)

  def test_audio_glob_compatibility(self):
    audio_glob_1 = source_info.AudioSourceConfig(
        dataset_name='pos',
        base_path='/foo',
        file_glob='*.wav',
        target_sample_rate_hz=16000,
    )
    with self.subTest('different_base_path'):
      glob_diff_base_path = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/bar',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
      )
      self.assertTrue(audio_glob_1.is_compatible(glob_diff_base_path))

    with self.subTest('different_dataset_name'):
      glob_diff_dataset_name = source_info.AudioSourceConfig(
          dataset_name='neg',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_dataset_name))

    with self.subTest('different_target_sample_rate_hz'):
      glob_diff_sample_rate = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=24000,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_sample_rate))

    with self.subTest('different_min_audio_len_s'):
      glob_diff_min_audio_len_s = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
          min_audio_len_s=2.0,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_min_audio_len_s))

  def test_audio_sources_merge_update(self):
    audio_sources_1 = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='pos',
                base_path=self.tempdir,
                file_glob='pos/*.wav',
            ),
            source_info.AudioSourceConfig(
                dataset_name='neg',
                base_path=self.tempdir,
                file_glob='neg/*.wav',
            ),
        )
    )
    with self.subTest('no_overlap'):
      disjoint = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='qua',
                  base_path=self.tempdir,
                  file_glob='qua/*.wav',
              ),
              source_info.AudioSourceConfig(
                  dataset_name='huh',
                  base_path=self.tempdir,
                  file_glob='huh/*.wav',
              ),
          )
      )
      got = audio_sources_1.merge_update(disjoint)
      self.assertLen(got.audio_globs, 4)

    with self.subTest('update_base_path'):
      overlap = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='pos',
                  base_path='/other/basedir',
                  file_glob='pos/*.wav',
              ),
          )
      )
      got = audio_sources_1.merge_update(overlap)
      self.assertLen(got.audio_globs, 2)
      self.assertEqual(got.audio_globs[0].base_path, '/other/basedir')

    with self.subTest('update_incompatible_audio_glob'):
      incompatible = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='pos',
                  base_path=self.tempdir,
                  file_glob='pos/*.wav',
                  target_sample_rate_hz=24000,
              ),
          )
      )
      with self.assertRaises(ValueError):
        audio_sources_1.merge_update(incompatible)


if __name__ == '__main__':
  absltest.main()
