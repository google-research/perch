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

"""Tests for xeno_canto."""

import functools
import shutil
import tempfile

from chirp.data import filter_scrub_utils as fsu
from chirp.train_tests import fake_dataset
import pandas as pd
import tensorflow_datasets as tfds

from absl.testing import absltest


class DataProcessingTest(absltest.TestCase):
  """Used to factorize the common setup between data processing tests."""

  def setUp(self):
    super().setUp()
    # We define a toy dataframe that is easy to manually check
    self.toy_df = pd.DataFrame({
        'species_code': ['ostric2', 'ostric3', 'grerhe1'],
        'Common name': ['Common Ostrich', 'Somali Ostrich', 'Greater Rhea'],
        'bg_labels': [
            ['ostric3', 'grerhe1'],
            ['ostric2', 'grerhe1'],
            ['ostric2', 'ostric3'],
        ],
        'Country': ['Colombia', 'Australia', 'France'],
    })

    # Additionally, we define a dataframe that mimics the actual dataset
    # of interest. Test will be carried out on both dataframes.
    self.temp_dir = tempfile.mkdtemp()
    # self.temp_dir = self.create_tempdir()
    # create_tempdir() raises an UnparsedFlagAccessError when debugged locally,
    # Using `tempdir` instead and manually deleting the folder after test.
    fake_builder = fake_dataset.FakeDataset(data_dir=self.temp_dir)
    fake_builder.download_and_prepare()
    self.fake_info = fake_builder.info
    self.fake_df = tfds.as_dataframe(fake_builder.as_dataset()['train'])

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.temp_dir)


class NotInTests(DataProcessingTest):

  def test_filtering_ideal(self):
    """Ensure filtering produces expected results in nominal case."""

    # 1) Tests on the toy dataFrame.
    test_df = self.toy_df.copy()
    fn_call = functools.partial(
        fsu.is_not_in, key='Country', values=['France', 'Brazil']
    )

    # Ensure we're properly filtering out species
    self.assertEqual(fn_call(test_df).tolist(), [True, True, False])

    # 2) Tests on the fake dataFrame.

    df = self.fake_df.copy()
    targeted_country = b'South Africa'
    # If that doesn't pass, maybe South Africa is just no longer a good example
    # to test on. Check fake_dataset how the recordings are populated.
    self.assertIn(targeted_country, df['country'].unique())
    filtering_fn = functools.partial(
        fsu.is_not_in, key='country', values=[targeted_country]
    )
    filtered_df = df[filtering_fn(df)]
    self.assertGreater(len(df), len(filtered_df))

    # Ensure we've filtered all South African species.
    self.assertEqual(filtered_df['country'].isin([targeted_country]).sum(), 0)

    # Ensure we did not filter too many countries.
    self.assertEqual(
        (df['country'] == targeted_country).sum(), len(df) - len(filtered_df)
    )

  def test_filtering_wrong_field(self):
    """Ensure filtering raises adequate error when the field does not exist."""
    test_df = self.toy_df.copy()

    # Make a query with an error in the name of the field
    with self.assertRaises(ValueError):
      _ = fsu.is_not_in(test_df, key='county', values=['France', 'Brazil'])

  def test_filtering_wrong_type(self):
    """Ensure filtering raises adequate error with wrong input type."""
    test_df = self.toy_df.copy()
    with self.assertRaises(TypeError):
      _ = (
          fsu.is_not_in(test_df, key='Country', values=[b'France', b'Brazil']),
      )


class ScrubTest(DataProcessingTest):

  def test_scrubbing_ideal(self):
    """Ensure scrubbing works as expected in nominal case."""
    expected_df = self.toy_df.copy()
    expected_df['bg_labels'] = [
        ['grerhe1'],
        ['ostric2', 'grerhe1'],
        ['ostric2'],
    ]

    # 1) Simple scrubbing queries on the toy dataframe
    test_df = self.toy_df.copy()
    test_df = test_df.apply(
        lambda row: fsu.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand',
    )
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

    # 2) Simple scrubbing queries on the fake_df
    df = self.fake_df.copy()
    targeted_country = b'South Africa'
    key = 'label'
    all_species = df[key].explode(key).unique()
    south_african_species = (
        df[df['country'] == targeted_country][key].explode(key).unique()
    )
    scrubbing_fn = functools.partial(
        fsu.scrub, key=key, values=south_african_species
    )
    scrubbed_df = df.apply(scrubbing_fn, axis=1, result_type='expand')

    remaining_species = scrubbed_df[key].explode(key).unique()
    self.assertEqual(
        set(south_african_species),
        set(all_species).difference(remaining_species),
    )

  def test_scrubbing_empty_col(self):
    """Ensure scrubbing doesn't do anything if the column is empty."""

    expected_df = self.toy_df.copy()
    test_df = self.toy_df.copy()
    expected_df['bg_labels'] = [[], [], []]
    test_df['bg_labels'] = [[], [], []]
    test_df = test_df.apply(
        lambda row: fsu.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand',
    )
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

  def test_scrubbing_empty_query(self):
    """Ensure scrubbing doesn't do anything if `values` is an empty list."""

    expected_df = self.toy_df.copy()
    test_df = self.toy_df.copy()
    test_df = test_df.apply(
        lambda row: fsu.scrub(row, 'bg_labels', []),
        axis=1,
        result_type='expand',
    )
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

  def test_scrub_no_side_effects(self):
    """Ensure scrubbing operation does not have side-effects."""
    df = self.fake_df.copy()
    key = 'label'
    # Scrub every foreground label.
    scrubbing_fn = functools.partial(
        fsu.scrub, key=key, values=df.explode(key)[key].unique()
    )
    _ = df.apply(scrubbing_fn, axis=1, result_type='expand')

    # If scrub function had side-effects, e.g. modified the row in-place,
    # the df would also change.
    self.assertTrue(self.fake_df.equals(df))


class QueryTest(DataProcessingTest):

  def test_masking_query(self):
    """Ensure masking queries work as expected."""

    mask_query = fsu.Query(
        op=fsu.MaskOp.IN, kwargs={'key': 'species_code', 'values': ['ostric2']}
    )
    self.assertEqual(
        fsu.apply_query(self.toy_df, mask_query).tolist(), [True, False, False]
    )

  def test_contains_query(self):
    mask_query = fsu.Query(
        op=fsu.MaskOp.CONTAINS_ANY,
        kwargs={'key': 'bg_labels', 'values': ['ostric2']},
    )
    self.assertEqual(
        fsu.apply_query(self.toy_df, mask_query).tolist(), [False, True, True]
    )

    mask_query = fsu.Query(
        op=fsu.MaskOp.CONTAINS_NO,
        kwargs={'key': 'bg_labels', 'values': ['ostric1', 'ostric2']},
    )
    self.assertEqual(
        fsu.apply_query(self.toy_df, mask_query).tolist(), [True, False, False]
    )

  def test_scrub_query(self):
    """Ensure scrubbing queries work as expected."""

    # Test scrubbing on list-typed fields.
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={'key': 'bg_labels', 'values': ['ostric2']},
    )
    df = fsu.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['bg_labels'] = [
        ['ostric3', 'grerhe1'],
        ['grerhe1'],
        ['ostric3'],
    ]

    # Test scrubbing on str-typed fields.
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={'key': 'species_code', 'values': ['ostric2', 'ostric3']},
    )
    df = fsu.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['species_code'] = ['', '', 'grerhe1']
    self.assertEqual(expected_df.to_dict(), df.to_dict())

    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB_ALL_BUT,
        kwargs={'key': 'species_code', 'values': ['ostric2', 'ostric3']},
    )
    df = fsu.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['species_code'] = ['ostric2', 'ostric3', '']
    self.assertEqual(expected_df.to_dict(), df.to_dict())

    # Test with a replace value
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={
            'key': 'species_code',
            'values': ['ostric2', 'ostric3'],
            'replace_value': 'unknown',
        },
    )
    df = fsu.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['species_code'] = ['unknown', 'unknown', 'grerhe1']
    self.assertEqual(expected_df.to_dict(), df.to_dict())

    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={
            'key': 'bg_labels',
            'values': ['ostric2', 'ostric3'],
            'replace_value': 'unknown',
        },
    )
    df = fsu.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['bg_labels'] = [
        ['unknown', 'grerhe1'],
        ['unknown', 'grerhe1'],
        ['unknown', 'unknown'],
    ]
    self.assertEqual(expected_df.to_dict(), df.to_dict())

  def test_complemented_query(self):
    df = self.toy_df.copy()
    df['unique_key'] = [0, 1, 2]

    # Test nominal case with scrubbing query. Scrubbing does not remove any
    # samples. Therefore check that setting complement to True returns an
    # empty df.
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={'key': 'bg_labels', 'values': ['ostric2']},
    )
    new_df = fsu.apply_complement(
        df, fsu.QueryComplement(scrub_query, 'unique_key')
    )
    self.assertEmpty(new_df)

    # Test nominal case with filtering query
    filter_query = fsu.Query(
        op=fsu.TransformOp.FILTER,
        kwargs={
            'mask_op': fsu.MaskOp.IN,
            'op_kwargs': {'key': 'species_code', 'values': ['ostric2']},
        },
    )
    self.assertEqual(
        fsu.apply_complement(
            df, fsu.QueryComplement(filter_query, 'unique_key')
        ).to_dict(),
        df.drop([0]).to_dict(),
    )

    # Test that when values don't uniquely define each recording, an error
    # is raised
    with self.assertRaises(ValueError):
      df['unique_key'] = [0, 1, 1]
      fsu.apply_complement(df, fsu.QueryComplement(filter_query, 'unique_key'))

  def test_append_query(self):
    new_row = {
        'bg_labels': 'ignore',
        'species_code': 'ignore',
        'Common name': 'ignore',
        'Country': 'ignore',
    }
    append_query = fsu.Query(fsu.TransformOp.APPEND, {'row': new_row})
    new_df = fsu.apply_query(self.toy_df, append_query)
    self.assertEqual(
        new_df.to_dict(),
        pd.concat(
            [self.toy_df, pd.Series(new_row)], ignore_index=True
        ).to_dict(),
    )

    # Append query with keys not matching the dataframe
    append_query = fsu.Query(fsu.TransformOp.APPEND, {'row': {'a': 'b'}})
    with self.assertRaises(ValueError):
      fsu.apply_query(self.toy_df, append_query)


class QueryParallelTest(DataProcessingTest):

  def test_merge_or(self):
    mask_query_1 = fsu.Query(
        fsu.MaskOp.IN, {'key': 'Country', 'values': ['Colombia']}
    )
    mask_query_2 = fsu.Query(
        fsu.MaskOp.IN, {'key': 'species_code', 'values': ['grerhe1']}
    )

    query_parallel = fsu.QueryParallel(
        [mask_query_1, mask_query_2], fsu.MergeStrategy.OR
    )
    mask = fsu.apply_parallel(self.toy_df, query_parallel)
    self.assertEqual(mask.tolist(), [True, False, True])
    self.assertIn('Colombia', self.toy_df[mask]['Country'].tolist())
    self.assertIn('grerhe1', self.toy_df[mask]['species_code'].tolist())

    # Ensure an error is raised if any element is not a boolean Series.
    with self.assertRaises(TypeError):
      fsu.or_series([
          self.toy_df['species_code'],
          self.toy_df['species_code'] == 'ostric2',
      ])

    # Ensure an error is raised if Series don't pertain to the same set of
    # recordings.
    with self.assertRaises(RuntimeError):
      fsu.or_series([
          self.toy_df.drop(0)['species_code'] == 'ostric2',
          self.toy_df['species_code'] == 'ostric2',
      ])

  def test_merge_and(self):
    mask_query_1 = fsu.Query(
        fsu.MaskOp.IN, {'key': 'Country', 'values': ['Colombia', 'France']}
    )
    mask_query_2 = fsu.Query(
        fsu.MaskOp.IN, {'key': 'species_code', 'values': ['grerhe1']}
    )

    query_parallel = fsu.QueryParallel(
        [mask_query_1, mask_query_2], fsu.MergeStrategy.AND
    )
    mask = fsu.apply_parallel(self.toy_df, query_parallel)
    self.assertEqual(mask.tolist(), [False, False, True])

    # Ensure an error is raised if any element is not a boolean Series.
    with self.assertRaises(RuntimeError):
      fsu.and_series([
          self.toy_df['species_code'],
          self.toy_df['species_code'] == 'ostric2',
      ])

    # Ensure an error is raised if Series don't pertain to the same set of
    # recordings.
    with self.assertRaises(RuntimeError):
      fsu.and_series([
          self.toy_df.drop(0)['species_code'] == 'ostric2',
          self.toy_df['species_code'] == 'ostric2',
      ])

  def test_merge_concat_no_duplicates(self):
    filter_query_1 = fsu.Query(
        fsu.TransformOp.FILTER,
        {
            'mask_op': fsu.MaskOp.IN,
            'op_kwargs': {'key': 'Country', 'values': ['Colombia']},
        },
    )
    filter_query_2 = fsu.Query(
        fsu.TransformOp.FILTER,
        {
            'mask_op': fsu.MaskOp.IN,
            'op_kwargs': {
                'key': 'Country',
                'values': ['Colombia', 'Australia'],
            },
        },
    )
    scrub_query = fsu.Query(
        fsu.TransformOp.SCRUB, {'key': 'bg_labels', 'values': ['ostric3']}
    )

    # First recording will be selected by both queries (i.e. duplicate). The
    # following ensures it only appears once in the result when using
    # CONCAT_NO_DUPLICATES
    query_parallel = fsu.QueryParallel(
        [filter_query_1, filter_query_2], fsu.MergeStrategy.CONCAT_NO_DUPLICATES
    )
    self.assertTrue(
        fsu.apply_parallel(self.toy_df, query_parallel).equals(
            self.toy_df.drop(2)
        )
    )

    # In the following, we also apply scrubbing in the second Query. This
    # scrubbing will modify the first recording, and therefore it shouldn't be
    # counted as a duplicate anymore. In the final df, we should find two
    # versions of the first recording (the original, and the scrubbed one).
    query_parallel = fsu.QueryParallel(
        [filter_query_1, fsu.QuerySequence([filter_query_2, scrub_query])],
        fsu.MergeStrategy.CONCAT_NO_DUPLICATES,
    )
    scrubbed_r0 = self.toy_df.copy().loc[0]
    scrubbed_r0['bg_labels'] = ['grerhe1']
    # Here we don't use assertEqual with the .to_dict() because .to_dict()
    # automatically removes duplicate indexes, making it impossible to know
    # if duplicates were removed because of our merging strategy or because
    # of .to_dict().
    self.assertTrue(
        fsu.apply_parallel(self.toy_df, query_parallel).equals(
            pd.concat([
                self.toy_df.loc[[0]],
                pd.DataFrame([scrubbed_r0, self.toy_df.loc[1]]),
            ])
        )
    )

    # Ensure the concatenation raises an error if the two dataframes don't have
    # the exact same columns.
    with self.assertRaises(RuntimeError):
      fsu.concat_no_duplicates(
          [self.toy_df, self.toy_df[['species_code', 'bg_labels']]]
      )


class QuerySequenceTest(DataProcessingTest):

  def test_untargeted_filter_scrub(self):
    """Ensure that applying a QuerySequence (no masking specified) works."""
    filter_args = {
        'mask_op': fsu.MaskOp.IN,
        'op_kwargs': {'key': 'species_code', 'values': ['ostric3', 'ostric2']},
    }
    filter_query = fsu.Query(op=fsu.TransformOp.FILTER, kwargs=filter_args)
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={'key': 'bg_labels', 'values': ['ostric2']},
    )
    query_sequence = fsu.QuerySequence(queries=[filter_query, scrub_query])
    df = fsu.apply_sequence(self.toy_df, query_sequence)
    expected_df = pd.DataFrame({
        'species_code': ['ostric2', 'ostric3'],
        'Common name': [
            'Common Ostrich',
            'Somali Ostrich',
        ],
        'bg_labels': [
            ['ostric3', 'grerhe1'],
            ['grerhe1'],
        ],
        'Country': ['Colombia', 'Australia'],
    })
    self.assertEqual(expected_df.to_dict(), df.to_dict())

  def test_targeted_filter_scrub(self):
    """Test QuerySequence on a subset of samples (w/ masking query)."""

    filter_args = {
        'mask_op': fsu.MaskOp.IN,
        'op_kwargs': {'key': 'species_code', 'values': ['ostric3', 'grerhe1']},
    }
    filter_query = fsu.Query(op=fsu.TransformOp.FILTER, kwargs=filter_args)
    scrub_query = fsu.Query(
        op=fsu.TransformOp.SCRUB,
        kwargs={'key': 'bg_labels', 'values': ['ostric2']},
    )
    query_sequence = fsu.QuerySequence(
        queries=[filter_query, scrub_query],
        mask_query=fsu.Query(
            op=fsu.MaskOp.IN,
            kwargs={'key': 'Country', 'values': ['Colombia', 'Australia']},
        ),
    )
    df = fsu.apply_sequence(self.toy_df, query_sequence)
    # In the example, only samples 1 and 3 have country values in
    # ['Colombia', 'Australia']. Therefore, sample 2 will not be affected at all
    # by any query. Sample 3 will be removed because of the first filtering
    # query. Sample 1 will survive the first filtering query, but will be
    # scrubbed out from its 'ostric2' bg_label.
    expected_df = pd.DataFrame({
        'species_code': ['ostric3', 'grerhe1'],
        'Common name': ['Somali Ostrich', 'Greater Rhea'],
        'bg_labels': [['grerhe1'], ['ostric2', 'ostric3']],
        'Country': ['Australia', 'France'],
    })
    self.assertEqual(
        expected_df.sort_values('species_code').to_dict('list'),
        df.sort_values('species_code').to_dict('list'),
    )

  def test_nested_query_sequence(self):
    filter_args = {
        'mask_op': fsu.MaskOp.IN,
        'op_kwargs': {'key': 'species_code', 'values': ['ostric3', 'grerhe1']},
    }
    filter_query = fsu.Query(fsu.TransformOp.FILTER, filter_args)
    mask_query = fsu.Query(
        fsu.MaskOp.IN, {'key': 'species_code', 'values': ['grerhe1']}
    )
    scrub_query = fsu.Query(
        fsu.TransformOp.SCRUB, {'key': 'bg_labels', 'values': ['ostric2']}
    )
    equivalent_queries = [
        fsu.QuerySequence(
            [filter_query, fsu.QuerySequence([scrub_query], mask_query)]
        ),
        fsu.QuerySequence(
            [
                filter_query,
                fsu.QuerySequence([filter_query]),
                fsu.QuerySequence([scrub_query], mask_query),
            ],
        ),
        fsu.QuerySequence(
            [
                fsu.QuerySequence([filter_query]),
                filter_query,
                fsu.QuerySequence([scrub_query], mask_query),
            ],
        ),
        fsu.QuerySequence(
            [
                fsu.QuerySequence([]),
                filter_query,
                fsu.QuerySequence([scrub_query], mask_query),
            ],
        ),
        fsu.QuerySequence(
            [
                filter_query,
                fsu.QuerySequence([filter_query, scrub_query], mask_query),
            ],
        ),
    ]
    expected_df = self.toy_df.drop(0)
    expected_df['bg_labels'] = [['ostric2', 'grerhe1'], ['ostric3']]
    for query_sequence in equivalent_queries:
      self.assertEqual(
          expected_df.to_dict(),
          fsu.apply_sequence(self.toy_df, query_sequence).to_dict(),
      )


class FilterByClasslistTest(DataProcessingTest):

  def test_filter_not_in_class_list(self):
    """Test filtering all items not in the class list ."""
    filter_query = fsu.filter_not_in_class_list('species_code', 'tiny_species')
    expected_df = pd.DataFrame({
        'species_code': ['ostric3', 'grerhe1'],
        'Common name': ['Somali Ostrich', 'Greater Rhea'],
        'bg_labels': [['ostric2', 'grerhe1'], ['ostric2', 'ostric3']],
        'Country': ['Australia', 'France'],
    })
    self.assertEqual(
        fsu.apply_query(self.toy_df, filter_query).values.tolist(),
        expected_df.values.tolist(),
    )

  def test_filter_in_class_list(self):
    """Test filtering all items not in the class list."""
    filter_query = fsu.filter_in_class_list('species_code', 'tiny_species')
    expected_df = pd.DataFrame({
        'species_code': ['ostric2'],
        'Common name': ['Common Ostrich'],
        'bg_labels': [
            ['ostric3', 'grerhe1'],
        ],
        'Country': ['Colombia'],
    })
    self.assertEqual(
        fsu.apply_query(self.toy_df, filter_query).values.tolist(),
        expected_df.values.tolist(),
    )

  # def test_filter_in_class_list(self):
  def test_filter_contains_no_class_list(self):
    """Test filtering all items not in target class list ."""
    filter_query = fsu.filter_contains_no_class_list(
        'bg_labels', 'tiny_species'
    )

    expected_df = pd.DataFrame({
        'species_code': ['ostric2'],
        'Common name': ['Common Ostrich'],
        'bg_labels': [['ostric3', 'grerhe1']],
        'Country': ['Colombia'],
    })
    self.assertEqual(
        fsu.apply_query(self.toy_df, filter_query).values.tolist(),
        expected_df.values.tolist(),
    )

  def test_filter_contains_any_class_list(self):
    """Test filtering any  items that is in target class list ."""
    filter_query = fsu.filter_contains_any_class_list(
        'bg_labels', 'tiny_species'
    )

    expected_df = pd.DataFrame({
        'species_code': ['ostric3', 'grerhe1'],
        'Common name': ['Somali Ostrich', 'Greater Rhea'],
        'bg_labels': [['ostric2', 'grerhe1'], ['ostric2', 'ostric3']],
        'Country': ['Australia', 'France'],
    })
    self.assertEqual(
        fsu.apply_query(self.toy_df, filter_query).values.tolist(),
        expected_df.values.tolist(),
    )


if __name__ == '__main__':
  absltest.main()
