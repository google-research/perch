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

from chirp.data.bird_taxonomy import bird_taxonomy
from etils import epath
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest


class BirdTaxonomyTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for the bird taxonomy dataset."""
  DATASET_CLASS = bird_taxonomy.BirdTaxonomy
  EXAMPLE_DIR = DATASET_CLASS.code_path.parent / 'placeholder_data'
  DL_EXTRACT_RESULT = {'taxonomy_info': 'taxonomy_info.json'}
  SPLITS = {'train': 4}
  SKIP_CHECKSUMS = True

  def setUp(self):
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

    _ = tfds.core.lazy_imports.librosa

    # We patch `self.DATASET_CLASS` _before_ calling `super().setUp()` because
    # `DatasetBuilderTestCase` instantiates `self.DATASET_CLASS` in its
    # `setUp()` method.
    metadata_patcher = mock.patch.object(self.DATASET_CLASS,
                                         '_load_taxonomy_metadata')
    mock_load_taxonomy_metadata = metadata_patcher.start()
    mock_load_taxonomy_metadata.return_value = pd.read_json(
        self.EXAMPLE_DIR /
        'taxonomy_info.json')[['species_code', 'genus', 'family', 'order']]

    url_patcher = mock.patch.object(self.DATASET_CLASS, 'GCS_URL',
                                    epath.Path(self.tempdir))
    url_patcher.start()
    subdir = epath.Path(self.tempdir) / 'audio-data' / 'fakecode1'
    subdir.mkdir(parents=True)
    for i in range(4):
      tfds.core.lazy_imports.pydub.AudioSegment.silent(duration=10000).export(
          subdir / f'XC{i:05d}.mp3', format='mp3')

    super().setUp()
    self.patchers.extend([metadata_patcher, url_patcher])

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)


class Int16AsFloatTensorTest(absltest.TestCase):
  """Tests for the Int16AsFloatTensor feature."""

  def test_encode_example(self):
    feature = bird_taxonomy.Int16AsFloatTensor(shape=[None])
    np.testing.assert_allclose(
        feature.encode_example([-1.0, 0.0]),
        np.array([-(2**15), 0], dtype=np.int16))

  def test_reconstruct(self):
    example_data = [-1.0, 0.0, 0.5]
    feature = bird_taxonomy.Int16AsFloatTensor(
        shape=[None], encoding=tfds.features.Encoding.ZLIB)
    np.testing.assert_allclose(
        example_data,
        feature.decode_example(feature.encode_example(example_data)))

  def test_exception_on_non_float(self):
    feature = bird_taxonomy.Int16AsFloatTensor(shape=[None])
    self.assertRaises(ValueError, feature.encode_example,
                      np.array([-1, 0, 0], dtype=np.int16))

  def test_exception_on_out_of_bound_values(self):
    feature = bird_taxonomy.Int16AsFloatTensor(shape=[None])
    self.assertRaises(ValueError, feature.encode_example, [1.0])
    self.assertRaises(ValueError, feature.encode_example, [-1.5])


if __name__ == '__main__':
  absltest.main()
