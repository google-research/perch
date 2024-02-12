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

"""soundevents dataset tests."""

import shutil
import tempfile
from unittest import mock

from chirp.data import soundevents
from etils import epath
import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest


def mock_localization_fn(audio, sr, interval_length_s, max_intervals):
  del audio
  del max_intervals
  target_length = sr * interval_length_s
  return [(0, target_length)]


class SoundeventsTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for the soundevents dataset."""

  DATASET_CLASS = soundevents.Soundevents
  BUILDER_CONFIG_NAMES_TO_TEST = [
      config.name
      for config in DATASET_CLASS.BUILDER_CONFIGS
      if config.name in ['fsd50k_full_length']
  ]
  DL_EXTRACT_RESULT = {}
  DL_SAMPLE_FILES = {
      'dev_samples': (
          DATASET_CLASS.code_path.parent
          / 'placeholder_data'
          / 'dev_samples.json'
      ),
      'eval_samples': (
          DATASET_CLASS.code_path.parent
          / 'placeholder_data'
          / 'eval_samples.json'
      ),
  }
  SKIP_CHECKSUMS = True

  @classmethod
  def setUpClass(cls):
    """"""
    super().setUpClass()

    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    cls.tempdir = tempfile.mkdtemp()

    _ = tfds.core.lazy_imports.librosa

    # soundevent uses FSD50K_DATASET_INFO to load train and test splits.
    # for creating test tfds, dev and eval set are created from sameples
    # create audio files for dev set placeholder data samples
    df_dev_samples = pd.read_json(cls.DL_SAMPLE_FILES['dev_samples'])
    df_dev_samples.columns = ['fname', 'labels', 'mids', 'split']
    subdir = epath.Path(cls.tempdir) / 'dev_audio'
    subdir.mkdir(parents=True)
    for _, row in df_dev_samples.iterrows():
      tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=10000).export(
          subdir / f'{row["fname"]}.wav', format='wav'
      )

    # create audio files for eval set from placeholder_data samples
    df_eval_samples = pd.read_json(cls.DL_SAMPLE_FILES['eval_samples'])
    df_eval_samples.columns = ['fname', 'labels', 'mids']
    subdir = epath.Path(cls.tempdir) / 'eval_audio'
    subdir.mkdir(parents=True)
    print(subdir)
    for _, row in df_eval_samples.iterrows():
      tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=10000).export(
          subdir / f'{row["fname"]}.wav', format='wav'
      )

    subdir = epath.Path(cls.tempdir) / 'FSD50K.ground_truth'
    subdir.mkdir(parents=True)
    df_dev_samples.to_csv(subdir / 'dev.csv', index=False)
    df_eval_samples.to_csv(subdir / 'eval.csv', index=False)
    cls.DL_EXTRACT_RESULT['dataset_info_dev'] = subdir / 'dev.csv'
    cls.DL_EXTRACT_RESULT['dataset_info_eval'] = subdir / 'eval.csv'
    cls.SPLITS = {'train': len(df_dev_samples), 'test': len(df_eval_samples)}
    cls.EXAMPLE_DIR = epath.Path(cls.tempdir)
    cls.url_patcher = mock.patch.object(
        cls.DATASET_CLASS, 'GCS_URL', epath.Path(cls.tempdir)
    )
    cls.url_patcher.start()
    mock_gcs_url = epath.Path(cls.tempdir)
    mock_dataset_config = {
        'dev': {
            'ground_truth_file': mock_gcs_url / 'FSD50K.ground_truth/dev.csv',
            'audio_dir': mock_gcs_url / 'dev_audio',
        },
        'eval': {
            'ground_truth_file': mock_gcs_url / 'FSD50K.ground_truth/eval.csv',
            'audio_dir': mock_gcs_url / 'eval_audio',
        },
    }
    cls.config_patcher = mock.patch.object(
        cls.DATASET_CLASS, 'DATASET_CONFIG', mock_dataset_config
    )
    cls.config_patcher.start()

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()

    cls.url_patcher.stop()
    cls.config_patcher.stop()
    shutil.rmtree(cls.tempdir)

  @absltest.skip
  def test_tags_are_valid(self):
    pass


if __name__ == '__main__':
  absltest.main()
