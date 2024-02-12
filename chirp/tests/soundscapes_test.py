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

"""Bird taxonomy dataset tests."""

import shutil
import tempfile
from unittest import mock

from chirp.data.soundscapes import soundscapes
from chirp.data.soundscapes import soundscapes_lib
from chirp.taxonomy import namespace
from etils import epath
import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest


def mock_localization_fn(audio, sr, interval_length_s, max_intervals):
  del audio
  del max_intervals
  target_length = sr * interval_length_s
  return [(0, target_length)]


class SoundscapeTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for the soundscape dataset."""

  DATASET_CLASS = soundscapes.Soundscapes
  BUILDER_CONFIG_NAMES_TO_TEST = [
      config.name
      for config in DATASET_CLASS.BUILDER_CONFIGS
      if config.name in ['caples']
  ]
  EXAMPLE_DIR = DATASET_CLASS.code_path.parent / 'placeholder_data'
  DL_EXTRACT_RESULT = {
      'segments': EXAMPLE_DIR / 'test.csv',
  }
  SPLITS = {'train': 3}
  SKIP_CHECKSUMS = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    cls.tempdir = tempfile.mkdtemp()

    _ = tfds.core.lazy_imports.librosa

    cls.metadata_patcher = mock.patch.object(soundscapes_lib, 'load_class_list')
    cls.loc_patcher = mock.patch.object(
        cls.DATASET_CLASS.BUILDER_CONFIGS[0],
        'localization_fn',
        mock_localization_fn,
    )
    cls.url_patcher = mock.patch.object(
        cls.DATASET_CLASS.BUILDER_CONFIGS[0],
        'audio_dir',
        epath.Path(cls.tempdir),
    )
    # We mock the localization part with a function that finds signal in the
    # first interval_length_s (5 sec.). This means that fake segments 1, 2 and 4
    # should be selected. Segment 3 should not be selected (not overlap with
    # the localization_fn output) and segment 5 should also be skipped because
    # the annotation is invalid (end < start). So we should end up with 3
    # recordings.
    cls.loc_patcher.start()
    mock_load_class_list = cls.metadata_patcher.start()
    mock_load_class_list.return_value = namespace.ClassList(
        'test_namespace',
        ['fakecode1', 'fakecode2', 'superrare', 'superfly'],
    )
    fake_segments = pd.read_csv(cls.EXAMPLE_DIR / 'test.csv')
    fake_segments['ebird_codes'] = fake_segments['ebird_codes'].apply(
        lambda codes: codes.split()
    )

    cls.url_patcher.start()
    subdir = epath.Path(cls.tempdir) / 'caples' / 'audio'
    subdir.mkdir(parents=True)
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_1.flac', format='flac'
    )
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_2.wav', format='wav'
    )
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_3.wav', format='wav'
    )
    tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=100000).export(
        subdir / 'soundscape_4.wav', format='wav'
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls.metadata_patcher.stop()
    cls.loc_patcher.stop()
    cls.url_patcher.stop()
    shutil.rmtree(cls.tempdir)

  # TODO(bartvm): Remove when tensorflow-datasets PyPI package is fixed
  @absltest.skip
  def test_tags_are_valid(self):
    pass


if __name__ == '__main__':
  absltest.main()
