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

from chirp.data import filter_scrub_utils as fsu
from chirp.data import tfds_features
from chirp.data.bird_taxonomy import bird_taxonomy
from etils import epath
import numpy as np
import tensorflow_datasets as tfds

from absl.testing import absltest


class BirdTaxonomyTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for the bird taxonomy dataset."""

  DATASET_CLASS = bird_taxonomy.BirdTaxonomy
  BUILDER_CONFIG_NAMES_TO_TEST = [
      config.name
      for config in DATASET_CLASS.BUILDER_CONFIGS
      if not (
          'tiny' in config.name
          or 'upstream' in config.name
          or 'downstream' in config.name
          or 'representative' in config.name
      )
  ]
  EXAMPLE_DIR = DATASET_CLASS.code_path.parent / 'placeholder_data'
  DL_EXTRACT_RESULT = {'taxonomy_info': 'taxonomy_info.json'}
  SPLITS = {'train': 4}
  SKIP_CHECKSUMS = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    cls.tempdir = tempfile.mkdtemp()

    _ = tfds.core.lazy_imports.librosa

    cls.url_patcher = mock.patch.object(
        cls.DATASET_CLASS, 'GCS_URL', epath.Path(cls.tempdir)
    )
    cls.query_patchers = []
    for i in [3, 4]:
      cls.query_patchers.append(
          mock.patch.object(
              cls.DATASET_CLASS.BUILDER_CONFIGS[i],
              'data_processing_query',
              fsu.QuerySequence([]),
          )
      )
    cls.url_patcher.start()
    for patcher in cls.query_patchers:
      patcher.start()
    subdir = epath.Path(cls.tempdir) / 'audio-data' / 'comter'
    subdir.mkdir(parents=True)
    for i in range(4):
      tfds.core.lazy_imports.pydub.AudioSegment(
          b'\0\1' * int(10_000 * 10),
          metadata={
              'channels': 1,
              'sample_width': 2,
              'frame_rate': 10_000,
              'frame_width': 2,
          },
      ).export(subdir / f'XC{i:05d}.mp3', format='mp3')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()

    cls.url_patcher.stop()
    for patcher in cls.query_patchers:
      patcher.stop()
    shutil.rmtree(cls.tempdir)

  # TODO(bartvm): Remove when tensorflow-datasets PyPI package is fixed
  @absltest.skip
  def test_tags_are_valid(self):
    pass


class Int16AsFloatTensorTest(absltest.TestCase):
  """Tests for the Int16AsFloatTensor feature."""

  def test_encode_example(self):
    feature = tfds_features.Int16AsFloatTensor(shape=[None], sample_rate=22050)
    np.testing.assert_allclose(
        feature.encode_example([-1.0, 0.0]),
        np.array([-(2**15), 0], dtype=np.int16),
    )

  def test_reconstruct(self):
    example_data = [-1.0, 0.0, 0.5]
    feature = tfds_features.Int16AsFloatTensor(
        shape=[None], sample_rate=22050, encoding=tfds.features.Encoding.ZLIB
    )
    np.testing.assert_allclose(
        example_data,
        feature.decode_example(feature.encode_example(example_data)),
    )

  def test_exception_on_non_float(self):
    feature = tfds_features.Int16AsFloatTensor(shape=[None], sample_rate=22050)
    self.assertRaises(
        ValueError, feature.encode_example, np.array([-1, 0, 0], dtype=np.int16)
    )

  def test_exception_on_out_of_bound_values(self):
    feature = tfds_features.Int16AsFloatTensor(shape=[None], sample_rate=22050)
    self.assertRaises(ValueError, feature.encode_example, [1.0])
    self.assertRaises(ValueError, feature.encode_example, [-1.5])


if __name__ == '__main__':
  absltest.main()
