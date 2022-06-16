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

"""Tests for xeno_canto."""

import functools

from chirp.data import filter_scrub_utils
import pandas as pd

from absl.testing import absltest
from absl.testing import parameterized


class FilteringScrubbingUnitTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.taxonomy_info = pd.DataFrame({
        'species_code': ['ostric2', 'ostric3', 'grerhe1'],
        'Common name': ['Common Ostrich', 'Somali Ostrich', 'Greater Rhea'],
        'bg_labels': [['ostric3', 'grerhe1'], ['ostric2', 'grerhe1'],
                      ['ostric2', 'ostric3']],
        'No.': [2, 2, 2],
        'Country': ['Colombia', 'Australia', 'France'],
        'No. Back': [1, 1, 1],
        'is_insect': [False, False, False],
        'no_species_code': [True, False, False],
        'to_verify': [False, True, False],
    })

  def test_filtering_ideal(self):
    """Ensure filtering produces expected results in nominal case."""
    # Ensure we're properly filtering out species
    expected_df = self.taxonomy_info.copy()
    expected_df = expected_df.drop([2])

    # A simple scrubbing query
    test_df = self.taxonomy_info.copy()
    fn_call = functools.partial(
        filter_scrub_utils.not_in, key='Country', values=['France', 'Brazil'])
    keep = test_df.apply(fn_call, axis=1, result_type='expand')
    test_df = test_df[keep]
    self.assertTrue(expected_df.equals(test_df))

  def test_filtering_wrong_queries(self):
    """Ensure filtering raises adequate errors when queries are not well formulated."""
    test_df = self.taxonomy_info.copy()

    # Make a query with an error in the name of the field
    with self.assertRaises(ValueError):
      test_df.apply(
          functools.partial(
              filter_scrub_utils.not_in,
              key='county',
              values=['France', 'Brazil']),
          axis=1,
          result_type='expand')

    # Make a query with values that don't have the same type as those in the
    # original dataframe.
    with self.assertRaises(TypeError):
      test_df.apply(
          functools.partial(
              filter_scrub_utils.not_in,
              key='Country',
              values=[b'France', b'Brazil']),
          axis=1,
          result_type='expand')

  def test_scrubbing_ideal(self):
    """Ensure we're properly filtering out species in nominal case."""
    expected_df = self.taxonomy_info.copy()
    expected_df['bg_labels'] = [['grerhe1'], ['ostric2', 'grerhe1'],
                                ['ostric2']]

    # A simple scrubbing query
    test_df = self.taxonomy_info.copy()
    test_df = test_df.apply(
        lambda row: filter_scrub_utils.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand')
    self.assertTrue(expected_df.equals(test_df))

    # Same but with all-but
    scrub_fn = functools.partial(
        filter_scrub_utils.scrub,
        key='bg_labels',
        values=['ostric3'],
        all_but=True)
    test_df = self.taxonomy_info.copy().apply(
        scrub_fn, axis=1, result_type='expand')
    expected_df['bg_labels'] = [['ostric3'], [], ['ostric3']]
    self.assertTrue(expected_df.equals(test_df))

  def test_scrubbing_empty_col(self):
    """Ensure scrubbing doesn't do anything if the values are an empty lists."""

    expected_df = self.taxonomy_info.copy()
    test_df = self.taxonomy_info.copy()
    expected_df['bg_labels'] = [[], [], []]
    test_df['bg_labels'] = [[], [], []]
    test_df = test_df.apply(
        lambda row: filter_scrub_utils.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand')
    self.assertTrue(expected_df.equals(test_df))


if __name__ == '__main__':
  absltest.main()
