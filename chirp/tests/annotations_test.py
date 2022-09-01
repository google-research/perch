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

"""Tests for annotations."""

import os

from chirp import path_utils
from chirp.taxonomy import annotations
from absl.testing import absltest


class AnnotationsTest(absltest.TestCase):

  def test_read_caples(self):
    dataset_path = path_utils.get_absolute_epath('tests/testdata/caples.csv')
    annos = annotations.read_caples_dataset_annotations(
        os.path.dirname(dataset_path))
    # There are six lines in the example file, but one contains a 'comros'
    # annotation which should be dropped.
    self.assertLen(annos, 5)

  def test_read_hawaii(self):
    # Grab a path to the testdata.
    dataset_path = os.path.dirname(
        path_utils.get_absolute_epath('tests/testdata/caples.csv'))
    annos = annotations.read_hawaii_dataset_annotations(dataset_path)
    # Check that only six annotations are kept, dropping the 'Spectrogram' views
    # which are redundant.
    self.assertLen(annos, 6)
    expected_labels = ['iiwi'] * 4 + ['omao', 'iiwi']
    for expected_label, anno in zip(expected_labels, annos):
      self.assertEqual(anno.filename, 'hawaii/hawaii_example.wav')
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])

  def test_read_ssw(self):
    dataset_path = path_utils.get_absolute_epath(
        'tests/testdata/ssw_annotations.csv')
    annos = annotations.read_ssw_dataset_annotations(dataset_path)
    self.assertLen(annos, 4)
    expected_labels = ['cangoo', 'blujay', 'rewbla', 'cangoo']
    for expected_label, anno in zip(expected_labels, annos):
      self.assertTrue(anno.filename.endswith('.flac'))
      self.assertEqual(anno.namespace, 'ebird2021')
      self.assertEqual(anno.label, [expected_label])


if __name__ == '__main__':
  absltest.main()
