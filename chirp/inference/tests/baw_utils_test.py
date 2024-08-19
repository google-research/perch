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

"""Tests for Bio-Acoustics Workbench utils."""

import os

from chirp.inference import baw_utils

from absl.testing import absltest

A2O_AUDIO_URL = 'https://api.acousticobservatory.org/audio_recordings'


class BawUtilsTest(absltest.TestCase):

  def test_basic_url_no_offset(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        '20210428T100000+1000_Five-Rivers-Dry-A_909057.flac', 0, 0
    )
    expected_url = os.path.join(A2O_AUDIO_URL, '909057/media.flac')
    self.assertEqual(url, expected_url)

  def test_url_with_offset_and_window_size(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        '/folder/site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav',
        10,
        30,
    )
    expected_url = os.path.join(
        A2O_AUDIO_URL, '909057/media.flac?start_offset=10&end_offset=40'
    )
    self.assertEqual(url, expected_url)

  def test_url_with_negative_offset(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_10000.wav', -10, 20
    )
    expected_url = os.path.join(A2O_AUDIO_URL, '10000/media.flac?end_offset=10')
    self.assertEqual(url, expected_url)

  def test_url_with_negative_window_size(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac', 5, -1
    )
    expected_url = os.path.join(
        A2O_AUDIO_URL, '909057/media.flac?start_offset=5'
    )
    self.assertEqual(url, expected_url)

  def test_url_with_zero_window_size(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav', 5, 0
    )
    expected_url = os.path.join(
        A2O_AUDIO_URL, '909057/media.flac?start_offset=5'
    )
    self.assertEqual(url, expected_url)

  def test_basic_url_no_offset_default_domain(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac', 0, 0
    )
    expected_url = os.path.join(A2O_AUDIO_URL, '909057/media.flac')
    self.assertEqual(url, expected_url)

  def test_basic_url_no_offset_custom_domain(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        '20210428T100000+1000_Five-Rivers-Dry-A_12345.flac',
        0,
        0,
        baw_domain='www.some.other.domain.com',
    )
    expected_url = (
        'https://www.some.other.domain.com/audio_recordings/12345/media.flac'
    )
    self.assertEqual(url, expected_url)

  def test_url_with_offset_and_window_size_custom_domain(self):
    url = baw_utils.make_baw_audio_url_from_file_id(
        'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav',
        10,
        30,
        baw_domain='example.com',
    )
    expected_url = (
        'https://example.com/audio_recordings/'
        '909057/media.flac?start_offset=10&end_offset=40'
    )
    self.assertEqual(url, expected_url)

  def test_invalid_file_id_format(self):
    with self.assertRaises(ValueError):
      baw_utils.make_baw_audio_url_from_file_id(
          'invalid_file_id', 10, 30, baw_domain='bad.domain.com'
      )


if __name__ == '__main__':
  absltest.main()
