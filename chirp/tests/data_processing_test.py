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
import shutil
import tempfile

from chirp.data import data_processing_ops as dpo
from chirp.tests import fake_dataset
import jax
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
        'bg_labels': [['ostric3', 'grerhe1'], ['ostric2', 'grerhe1'],
                      ['ostric2', 'ostric3']],
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
        dpo.is_not_in, key='Country', values=['France', 'Brazil'])

    # Ensure we're properly filtering out species
    self.assertEqual(
        test_df.apply(fn_call, axis=1, result_type='expand').tolist(),
        [True, True, False])

    # 2) Tests on the fake dataFrame.

    df = self.fake_df.copy()
    targeted_country = b'South Africa'
    # If that doesn't pass, maybe South Africa is just no longer a good example
    # to test on. Check fake_dataset how the recordings are populated.
    self.assertIn(targeted_country, df['country'].unique())
    filtering_fn = functools.partial(
        dpo.is_not_in, key='country', values=[targeted_country])
    filtered_df = df[df.apply(filtering_fn, axis=1, result_type='expand')]
    self.assertGreater(len(df), len(filtered_df))

    # Ensure we've filtered all South African species.
    self.assertEqual(filtered_df['country'].isin([targeted_country]).sum(), 0)

    # Ensure we did not filter too many countries.
    self.assertEqual((df['country'] == targeted_country).sum(),
                     len(df) - len(filtered_df))

  def test_filtering_wrong_field(self):
    """Ensure filtering raises adequate error when the field does not exist."""
    test_df = self.toy_df.copy()

    # Make a query with an error in the name of the field
    with self.assertRaises(ValueError):
      test_df.apply(
          functools.partial(
              dpo.is_not_in, key='county', values=['France', 'Brazil']),
          axis=1,
          result_type='expand')

  def test_filtering_wrong_type(self):
    """Ensure filtering raises adequate error with wrong input type."""
    test_df = self.toy_df.copy()
    with self.assertRaises(TypeError):
      test_df.apply(
          functools.partial(
              dpo.is_not_in, key='Country', values=[b'France', b'Brazil']),
          axis=1,
          result_type='expand')


class SamplingTest(DataProcessingTest):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.mkdtemp()
    fake_builder = fake_dataset.FakeDataset(data_dir=self.temp_dir)
    fake_builder.download_and_prepare()

    # We instead load a reduced_version, where the label space is smaller.
    # Otherwise, with only 100 samples, we end up with 1 sample per class,
    # which is too edgy to test sampling functionalities.
    self.fake_df = tfds.as_dataframe(fake_builder.as_dataset()['train_reduced'])

  def test_sample_groups(self):
    df = self.fake_df.copy()
    seed = jax.random.PRNGKey(0)
    n_classes = 2
    # Testing for two fields whose values have different types.
    for group_key in ['label', 'country']:
      new_df = df[dpo.sample_groups(df, n_classes, group_key, seed)]
      # Ensure that we end up with at least n_classes distinct classes.
      # If the recordings belong to multiple classes (e.g. they have multiple
      # foreground labels), then we may get more unique classes.
      self.assertGreaterEqual(
          len(new_df[group_key].explode(group_key).unique()), n_classes)

  def test_sample_recordings_per_group(self):
    df = self.fake_df.copy()
    # Testing for two fields whose values have different types.
    for group_key in ['label', 'country']:
      seed = jax.random.PRNGKey(0)

      all_classes = df[group_key].explode(group_key).unique()

      # Compute the minimum number of samples per class
      min_samples_per_class = min([
          df[group_key].map(lambda values: class_ in values).sum()
          for class_ in all_classes
      ])

      # Subsample such that we're sure all classes contain enough sample
      new_df = df[dpo.sample_recordings_per_group(df, min_samples_per_class,
                                                  group_key, seed)]

      # Ensure that we still have the same number of total classes.
      self.assertEqual(
          len(new_df[group_key].explode(group_key).unique()), len(all_classes))

      # Ensure that if we try to sample more than existing, a ValueError is
      # raised.
      with self.assertRaises(ValueError):
        dpo.sample_recordings_per_group(df, min_samples_per_class + 1,
                                        group_key, seed)

      # Ensure that each class has, at least, min_samples_per_class. Again, it
      # could have more, because each sample may have multiple foreground
      # labels.
      for class_ in all_classes:

        def test_in(value, class_=class_):
          return class_ in value

        self.assertGreaterEqual(new_df[group_key].map(test_in).sum(),
                                min_samples_per_class)

      # Ensure that if allow_overlap=False, we retrieve exactly
      # n_samples_per_class * n_initial_classes samples.
      # Here we test with a small number (5) to make sure
      # we'll be able to find 5 distinct samples for each class.
      n_samples_per_class = 5
      chosen_samples = dpo.sample_recordings_per_group(
          df, n_samples_per_class, group_key, seed, allow_overlap=False)
      self.assertEqual(
          chosen_samples.sum(),
          len(df[group_key].explode(group_key).unique()) * n_samples_per_class)

  def test_reproducibility(self):
    """Ensure that we can repeat the exact same sampling."""
    seed = jax.random.PRNGKey(0)
    n_classes = 5
    group_key = 'label'
    first_mask = dpo.sample_groups(self.fake_df, n_classes, group_key, seed)
    for _ in range(10):
      self.assertEqual(
          first_mask.tolist(),
          dpo.sample_groups(self.fake_df, n_classes, group_key, seed).tolist())


class ScrubTest(DataProcessingTest):

  def test_scrubbing_ideal(self):
    """Ensure scrubbing works as expected in nominal case."""
    expected_df = self.toy_df.copy()
    expected_df['bg_labels'] = [['grerhe1'], ['ostric2', 'grerhe1'],
                                ['ostric2']]

    # 1) Simple scrubbing queries on the toy dataframe
    test_df = self.toy_df.copy()
    test_df = test_df.apply(
        lambda row: dpo.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand')
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

    # 2) Simple scrubbing queries on the fake_df
    df = self.fake_df.copy()
    targeted_country = b'South Africa'
    key = 'label'
    all_species = df[key].explode(key).unique()
    south_african_species = df[df['country'] == targeted_country][key].explode(
        key).unique()
    scrubbing_fn = functools.partial(
        dpo.scrub, key=key, values=south_african_species)
    scrubbed_df = df.apply(scrubbing_fn, axis=1, result_type='expand')

    remaining_species = scrubbed_df[key].explode(key).unique()
    self.assertEqual(
        set(south_african_species),
        set(all_species).difference(remaining_species))

  def test_scrubbing_empty_col(self):
    """Ensure scrubbing doesn't do anything if the column is empty."""

    expected_df = self.toy_df.copy()
    test_df = self.toy_df.copy()
    expected_df['bg_labels'] = [[], [], []]
    test_df['bg_labels'] = [[], [], []]
    test_df = test_df.apply(
        lambda row: dpo.scrub(row, 'bg_labels', ['ostric3']),
        axis=1,
        result_type='expand')
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

  def test_scrubbing_empty_query(self):
    """Ensure scrubbing doesn't do anything if `values` is an empty list."""

    expected_df = self.toy_df.copy()
    test_df = self.toy_df.copy()
    test_df = test_df.apply(
        lambda row: dpo.scrub(row, 'bg_labels', []),
        axis=1,
        result_type='expand')
    self.assertEqual(expected_df.to_dict(), test_df.to_dict())

  def test_scrub_no_side_effects(self):
    """Ensure scrubbing operation does not have side-effects."""
    df = self.fake_df.copy()
    key = 'label'
    # Scrub every foreground label.
    scrubbing_fn = functools.partial(
        dpo.scrub, key=key, values=df.explode(key)[key].unique())
    _ = df.apply(scrubbing_fn, axis=1, result_type='expand')

    # If scrub function had side-effects, e.g. modified the row in-place,
    # the df would also change.
    self.assertTrue(self.fake_df.equals(df))


class QueryTest(DataProcessingTest):

  def test_masking_query(self):
    """Ensure masking queries (and completement) work as expected."""

    # Test mask query and complement
    mask_query = dpo.Query(
        op=dpo.MaskOp.IN, kwargs={
            'key': 'species_code',
            'values': ['ostric2']
        })
    self.assertEqual(
        dpo.apply_query(self.toy_df, mask_query).tolist(), [True, False, False])
    mask_query = dpo.Query(
        op=dpo.MaskOp.IN,
        complement=True,
        kwargs={
            'key': 'species_code',
            'values': ['ostric2']
        })
    self.assertEqual(
        dpo.apply_query(self.toy_df, mask_query).tolist(), [False, True, True])

  def test_scrub_query(self):
    """Ensure transform queries work as expected."""

    scrub_query = dpo.Query(
        op=dpo.TransformOp.SCRUB,
        kwargs={
            'key': 'bg_labels',
            'values': ['ostric2']
        })
    df = dpo.apply_query(self.toy_df, scrub_query)
    expected_df = self.toy_df.copy()
    expected_df['bg_labels'] = [['ostric3', 'grerhe1'], ['grerhe1'],
                                ['ostric3']]
    self.assertEqual(expected_df.to_dict(), df.to_dict())
    # Ensure that setting complement to True for a transform query raises an
    # error
    scrub_query = dpo.Query(
        op=dpo.TransformOp.SCRUB,
        complement=True,
        kwargs={
            'key': 'bg_labels',
            'values': ['ostric2']
        })
    with self.assertRaises(ValueError):
      dpo.apply_query(self.toy_df, scrub_query)

  def test_sampling_queries(self):
    """Simple sampling query. Just to ensure the SAMPLE_CLASSES op works ok."""

    filter_args = {
        'mask_op': dpo.MaskOp.SAMPLE_CLASSES,
        'op_kwargs': {
            'group_key': 'species_code',
            'n_groups': 2,
            'seed': jax.random.PRNGKey(0),
        }
    }
    filter_query = dpo.Query(op=dpo.TransformOp.FILTER, kwargs=filter_args)
    df = dpo.apply_query(self.toy_df, filter_query)
    # Test reproducibility below
    expected_df = pd.DataFrame({
        'species_code': ['ostric3', 'grerhe1'],
        'Common name': [
            'Somali Ostrich',
            'Greater Rhea',
        ],
        'bg_labels': [
            ['ostric2', 'grerhe1'],
            ['ostric2', 'ostric3'],
        ],
        'Country': ['Australia', 'France'],
    })
    self.assertEqual(expected_df.to_dict('list'), df.to_dict('list'))


class QuerySequenceTest(DataProcessingTest):

  def test_untargeted_filter_scrub(self):
    """Ensure that applying a QuerySequence (no masking specified) works."""
    filter_args = {
        'mask_op': dpo.MaskOp.IN,
        'op_kwargs': {
            'key': 'species_code',
            'values': ['ostric3', 'ostric2']
        },
    }
    filter_query = dpo.Query(op=dpo.TransformOp.FILTER, kwargs=filter_args)
    scrub_query = dpo.Query(
        op=dpo.TransformOp.SCRUB,
        kwargs={
            'key': 'bg_labels',
            'values': ['ostric2']
        })
    query_sequence = dpo.QuerySequence(queries=[filter_query, scrub_query])
    df = dpo.apply_sequence(self.toy_df, query_sequence)
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
        'mask_op': dpo.MaskOp.IN,
        'op_kwargs': {
            'key': 'species_code',
            'values': ['ostric3', 'grerhe1']
        }
    }
    filter_query = dpo.Query(op=dpo.TransformOp.FILTER, kwargs=filter_args)
    scrub_query = dpo.Query(
        op=dpo.TransformOp.SCRUB,
        kwargs={
            'key': 'bg_labels',
            'values': ['ostric2']
        })
    query_sequence = dpo.QuerySequence(
        queries=[filter_query, scrub_query],
        mask_query=dpo.Query(
            op=dpo.MaskOp.IN,
            kwargs={
                'key': 'Country',
                'values': ['Colombia', 'Australia']
            }))
    df = dpo.apply_sequence(self.toy_df, query_sequence)
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
        df.sort_values('species_code').to_dict('list'))


if __name__ == '__main__':
  absltest.main()
