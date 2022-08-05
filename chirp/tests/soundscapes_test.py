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

"""Bird taxonomy dataset tests."""

import shutil
import tempfile
from unittest import mock

from chirp.data.soundscapes import soundscapes
from etils import epath
import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest


class SoundscapeTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for the bird taxonomy dataset."""
  DATASET_CLASS = soundscapes.Soundscapes
  BUILDER_CONFIG_NAMES_TO_TEST = [
      config.name
      for config in DATASET_CLASS.BUILDER_CONFIGS
      if config.name in ['caples']
  ]
  EXAMPLE_DIR = DATASET_CLASS.code_path.parent / 'placeholder_data'
  SPLITS = {'train': 3}
  SKIP_CHECKSUMS = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    cls.tempdir = tempfile.mkdtemp()

    _ = tfds.core.lazy_imports.librosa

    cls.metadata_patcher = mock.patch.object(cls.DATASET_CLASS,
                                             '_load_taxonomy_metadata')
    cls.segments_patcher = mock.patch.object(cls.DATASET_CLASS,
                                             '_load_segments')
    mock_load_segments = cls.segments_patcher.start()
    mock_load_taxonomy_metadata = cls.metadata_patcher.start()
    mock_load_taxonomy_metadata.return_value = pd.read_json(
        cls.EXAMPLE_DIR / 'taxonomy_info.json')
    fake_segments = pd.read_csv(cls.EXAMPLE_DIR / 'test.csv')
    fake_segments['ebird_codes'] = fake_segments['ebird_codes'].apply(
        lambda codes: codes.split())
    mock_load_segments.return_value = fake_segments

    cls.url_patcher = mock.patch.object(cls.DATASET_CLASS, 'GCS_URL',
                                        epath.Path(cls.tempdir))

    cls.url_patcher.start()
    # _ = [patcher.start() for patcher in cls.config_patcher]
    subdir = epath.Path(cls.tempdir) / 'caples' / 'audio'
    subdir.mkdir(parents=True)
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_1.flac', format='flac')
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_2.wav', format='wav')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls.segments_patcher.stop()
    cls.metadata_patcher.stop()
    # _ = [patcher.stop() for patcher in cls.config_patcher]
    cls.url_patcher.stop()
    shutil.rmtree(cls.tempdir)


if __name__ == '__main__':
  absltest.main()
