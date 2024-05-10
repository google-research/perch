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

import os

from chirp.data import tfds_builder
import tensorflow_datasets as tfds

from absl.testing import absltest


def _manual_data_dir() -> str:
  return os.path.join(
      os.path.normpath(os.path.dirname(os.path.dirname(__file__))),
      'tests',
      'testdata',
      'tfds_builder_wav_directory_test',
  )


class FakeWavDirectory(tfds_builder.WavDirectoryBuilder):
  """Test-only concrete subclass of the abstract base class under test."""

  # A workaround for the following error in importlib_resources:
  # 'NoneType' object has no attribute 'submodule_search_locations'
  __module__ = tfds_builder.__name__

  VERSION = tfds.core.Version('1.2.3')
  RELEASE_NOTES = {
      '1.2.3': 'Initial release.',
  }

  def _description(self) -> str:
    return 'Unit test fixture with WAV files in a nested directory structure.'

  def _citation(self) -> str:
    return 'FakeWavDirectory, private communication, October, 2022.'


class WavDirectoryDatasetUnfilteredTest(tfds.testing.DatasetBuilderTestCase):
  """Tests WavDirectoryDataset with "unfiltered" configuration."""

  DATASET_CLASS = FakeWavDirectory
  EXAMPLE_DIR = _manual_data_dir()
  BUILDER_CONFIG_NAMES_TO_TEST = ['unfiltered']
  SPLITS = {'train': 3}

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  # TODO(bartvm): Remove when tensorflow-datasets PyPI package is fixed
  @absltest.skip
  def test_tags_are_valid(self):
    pass


class WavDirectoryDatasetSlicePeakedTest(tfds.testing.DatasetBuilderTestCase):
  """Tests WavDirectoryDataset with "slice_peaked" configuration."""

  DATASET_CLASS = FakeWavDirectory
  EXAMPLE_DIR = _manual_data_dir()
  BUILDER_CONFIG_NAMES_TO_TEST = ['slice_peaked']
  SPLITS = {'train': 2}

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  # TODO(bartvm): Remove when tensorflow-datasets PyPI package is fixed
  @absltest.skip
  def test_tags_are_valid(self):
    pass


if __name__ == '__main__':
  tfds.testing.test_main()
