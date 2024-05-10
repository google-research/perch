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

"""Tests for eval_lib."""

import functools
import os
import shutil
import tempfile
from typing import Any, Sequence

from chirp import config_utils
from chirp.configs import baseline_mel_conformer
from chirp.configs import config_globals
from chirp.data.bird_taxonomy import bird_taxonomy
from chirp.eval import callbacks
from chirp.eval import eval_lib
from chirp.taxonomy import namespace_db
from chirp.train_tests import fake_dataset
from chirp.train import classifier
import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

from absl.testing import absltest

_c = config_utils.callable_config


def _stub_localization_fn(
    audio: Any,
    sample_rate_hz: int,
    interval_length_s: float = 6.0,
    max_intervals: int = 5,
) -> Sequence[tuple[int, int]]:
  # The only purpose of this stub function is to avoid a default
  # `localization_fn` value of None in `BirdTaxonomyConfig` so that the audio
  # feature shape gets computed properly.
  del audio, sample_rate_hz, interval_length_s, max_intervals
  return []


class FakeBirdTaxonomy(fake_dataset.FakeDataset):
  BUILDER_CONFIGS = [
      bird_taxonomy.BirdTaxonomyConfig(
          name='fake_variant_1',
          localization_fn=_stub_localization_fn,
          interval_length_s=6.0,
      ),
      bird_taxonomy.BirdTaxonomyConfig(
          name='fake_variant_2',
          localization_fn=_stub_localization_fn,
          interval_length_s=6.0,
      ),
  ]


class LoadEvalDatasetsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.data_dir = tempfile.TemporaryDirectory('data_dir').name
    FakeBirdTaxonomy(
        data_dir=self.data_dir, config='fake_variant_1'
    ).download_and_prepare()
    FakeBirdTaxonomy(
        data_dir=self.data_dir, config='fake_variant_2'
    ).download_and_prepare()

  def test_return_value_structure(self):
    fake_config = ml_collections.ConfigDict()
    fake_config.dataset_configs = {
        'fake_dataset_1': {
            'tfds_name': 'fake_bird_taxonomy/fake_variant_1',
            'tfds_data_dir': self.data_dir,
            'pipeline': _c(
                'pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]
            ),
            'split': 'train',
        },
        'fake_dataset_2': {
            'tfds_name': 'fake_bird_taxonomy/fake_variant_2',
            'tfds_data_dir': self.data_dir,
            'pipeline': _c(
                'pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]
            ),
            'split': 'train',
        },
    }
    fake_config = config_utils.parse_config(
        fake_config, config_globals.get_globals()
    )
    eval_datasets = eval_lib.load_eval_datasets(fake_config)

    self.assertSameElements(
        ['fake_dataset_1', 'fake_dataset_2'], eval_datasets.keys()
    )
    for dataset in eval_datasets.values():
      self.assertIsInstance(dataset, tf.data.Dataset)
      self.assertContainsSubset(
          ['audio', 'label', 'bg_labels'], dataset.element_spec.keys()
      )

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.data_dir)


class GetEmbeddingsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.data_dir = tempfile.TemporaryDirectory('data_dir').name
    FakeBirdTaxonomy(
        data_dir=self.data_dir, config='fake_variant_1'
    ).download_and_prepare()

  def test_get_embeddings(self):
    fake_config = ml_collections.ConfigDict()
    fake_config.dataset_configs = {
        'fake_dataset_1': {
            'tfds_name': 'fake_bird_taxonomy/fake_variant_1',
            'tfds_data_dir': self.data_dir,
            'pipeline': _c(
                'pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]
            ),
            'split': 'train',
        },
    }
    fake_config.model_callback = lambda x: x + 1
    fake_config = config_utils.parse_config(
        fake_config, config_globals.get_globals()
    )
    dataset = eval_lib.load_eval_datasets(fake_config)
    (dataset_name,) = dataset.keys()
    dataset = dataset[dataset_name]
    embedded_dataset = eval_lib.get_embeddings(
        dataset, fake_config.model_callback, batch_size=1
    )
    self.assertContainsSubset(
        ['embedding'], embedded_dataset.element_spec.keys()
    )

    embedding = next(embedded_dataset.as_numpy_iterator())['embedding']
    self.assertTrue(((0 <= embedding) & (embedding <= 2)).all())

  def test_embedding_model_callback(self):
    placeholder_callback = callbacks.EmbeddingModelCallback(
        'placeholder_model', ml_collections.ConfigDict({'sample_rate': 32000})
    )
    fake_config = ml_collections.ConfigDict()
    fake_config.dataset_configs = {
        'fake_dataset_1': {
            'tfds_name': 'fake_bird_taxonomy/fake_variant_1',
            'tfds_data_dir': self.data_dir,
            'pipeline': _c(
                'pipeline.Pipeline', ops=[_c('pipeline.OnlyJaxTypes')]
            ),
            'split': 'train',
        },
    }
    fake_config.model_callback = lambda x: x + 1
    fake_config = config_utils.parse_config(
        fake_config, config_globals.get_globals()
    )
    dataset = eval_lib.load_eval_datasets(fake_config)
    (dataset_name,) = dataset.keys()
    dataset = dataset[dataset_name]
    embedded_dataset = eval_lib.get_embeddings(
        dataset, placeholder_callback.model_callback, batch_size=1
    )
    self.assertContainsSubset(
        ['embedding'], embedded_dataset.element_spec.keys()
    )

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.data_dir)


class EvalSetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.class_names = ['a', 'b', 'c']
    self.fake_embeddings_df = pd.DataFrame({
        'label': (['a'] * 4 + ['b'] * 4 + ['c'] * 4) * 2,
        'embedding': [[0.0]] * 24,
        'bg_labels': [
            '',
            'b',
            'c',
            'b c',
            '',
            'a',
            'c',
            'a c',
            '',
            'a',
            'b',
            'a b',
        ] * 2,
        'dataset_name': ['dataset_1'] * 12 + ['dataset_2'] * 12,
    })
    self.embedded_datasets = {
        'dataset_1': tf.data.Dataset.from_tensor_slices(
            self.fake_embeddings_df.groupby('dataset_name')
            .get_group('dataset_1')
            .to_dict('list')
        ),
        'dataset_2': tf.data.Dataset.from_tensor_slices(
            self.fake_embeddings_df.groupby('dataset_name')
            .get_group('dataset_2')
            .to_dict('list')
        ),
    }

  def test_prepare_eval_sets(self):
    partial_specification = functools.partial(
        eval_lib.EvalSetSpecification,
        class_names=self.class_names,
        search_corpus_classwise_mask_fn=(lambda _: 'label.str.contains("")'),
        class_representative_global_mask_expr='label.str.contains("")',
        class_representative_classwise_mask_fn=(
            lambda class_name: f'label.str.contains("{class_name}")'
        ),
        num_representatives_per_class=0,
    )

    fake_config = ml_collections.ConfigDict()
    fake_config.rng_seed = 1234
    fake_config.model_callback = {'learned_representations': {'a': [0.0]}}
    fake_config.debug = {'embedded_dataset_cache_path': ''}
    fake_config.eval_set_specifications = {
        'fake_specification_1': partial_specification(
            search_corpus_global_mask_expr='dataset_name == "dataset_1"'
        ),
        'fake_specification_2': partial_specification(
            search_corpus_global_mask_expr='dataset_name == "dataset_2"'
        ),
    }

    eval_sets = list(
        eval_lib.prepare_eval_sets(fake_config, self.embedded_datasets)
    )

    # There should be two eval sets.
    self.assertEqual(
        [eval_set.name for eval_set in eval_sets],
        ['fake_specification_1', 'fake_specification_2'],
    )
    # There should be one classwise eval set per class.
    for eval_set in eval_sets:
      class_names = [
          classwise_eval_set.class_name
          for classwise_eval_set in eval_set.classwise_eval_sets
      ]
      self.assertEqual(class_names, ['a', 'b', 'c'])

  def test_eval_set_generator(self):
    num_representatives_per_class = 2

    fake_config = ml_collections.ConfigDict()
    fake_config.rng_seed = 1234
    fake_config.model_callback = {'learned_representations': {'a': [0.0]}}
    fake_config.debug = {'embedded_dataset_cache_path': ''}
    fake_config.eval_set_specifications = {
        'fake_specification': eval_lib.EvalSetSpecification(
            search_corpus_global_mask_expr='dataset_name == "dataset_1"',
            class_names=self.class_names,
            search_corpus_classwise_mask_fn=(
                lambda n: f'not bg_labels.str.contains("{n}")'
            ),
            class_representative_global_mask_expr='dataset_name == "dataset_1"',
            class_representative_classwise_mask_fn=(
                lambda n: f'label.str.contains("{n}")'
            ),
            num_representatives_per_class=num_representatives_per_class,
        )
    }

    (eval_set,) = eval_lib.prepare_eval_sets(
        fake_config, self.embedded_datasets
    )
    for classwise_eval_set in eval_set.classwise_eval_sets:
      class_name = classwise_eval_set.class_name
      class_representatives_df = classwise_eval_set.class_representatives_df
      search_corpus_df = eval_set.search_corpus_df[
          classwise_eval_set.search_corpus_mask
      ]

      # We should get the number of class representatives we requested.
      self.assertLen(class_representatives_df, num_representatives_per_class)
      # All class representatives should have the label `class_name`.
      self.assertTrue((class_representatives_df['label'] == class_name).all())
      # According to our `search_corpus_classwise_mask_fn`, `class_name` should
      # not appear in any background label.
      self.assertTrue(
          (~search_corpus_df['bg_labels'].str.contains(class_name)).all()
      )
      # Class representatives should not be included in the search corpus.
      self.assertTrue(
          (~search_corpus_df.index.isin(class_representatives_df.index)).all()
      )
      # Embeddings from 'dataset_2' should not be found anywhere.
      self.assertTrue(
          (class_representatives_df['dataset_name'] != 'dataset_2').all()
      )
      self.assertTrue((search_corpus_df['dataset_name'] != 'dataset_2').all())
      # By construction of `self.embeddings_df`, we know that the above three
      # result in 4 + 2 + 12 = 18 rows being excluded.
      self.assertLen(search_corpus_df, 6)


class SearchProcedureTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.query = [1, 0]
    self.species_id = 'C'
    self.search_corpus = pd.DataFrame({
        'embedding': [[0, 1], [1, 0], [1, 1]],
        'label': ['B', 'A', 'C'],
        'bg_labels': ['B C', ' ', 'A'],
    })

  def test_query_search(self):
    score = pd.Series([0.0, 1.0, 0.7071])
    search_corpus_mask = pd.Series([False, True, True])

    actual_query_result = eval_lib._make_species_scores_df(
        score=score,
        species_id=self.species_id,
        search_corpus=self.search_corpus,
        search_corpus_mask=search_corpus_mask,
    )
    expected_query_result = pd.DataFrame({
        'score': score.tolist(),
        'species_match': [1, 0, 1],
        'label_mask': search_corpus_mask.tolist(),
    })
    actual_query_scores = actual_query_result['score'].round(4)
    expected_query_scores = expected_query_result['score']
    self.assertTrue((actual_query_scores == expected_query_scores).all())

    actual_query_matches = actual_query_result['species_match']
    expected_query_matches = expected_query_result['species_match']
    self.assertTrue((actual_query_matches == expected_query_matches).all())

    actual_label_mask = actual_query_result['label_mask']
    expected_label_mask = expected_query_result['label_mask']
    self.assertTrue((actual_label_mask == expected_label_mask).all())


class DefaultFunctionsTest(absltest.TestCase):

  def test_create_averaged_query(self):
    embedding1 = np.arange(0, 5)
    embedding2 = np.arange(1, 6)
    embeddings = [embedding1, embedding2]
    actual_avg_query = eval_lib.create_averaged_query(embeddings)
    expected_avg_query = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    self.assertTrue(actual_avg_query.tolist(), expected_avg_query.tolist())

  def test_cosine_similarity(self):
    embedding = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    actual_similarity = eval_lib.cosine_similarity(embedding, embedding)
    expected_similarity = 1.0
    np.testing.assert_allclose(actual_similarity, expected_similarity)

    orthog_embedding0 = np.array([[-0.5, 0.0, -0.5, 0.0, -0.5]])
    orthog_embedding1 = np.array([[0.0, 0.5, 0.0, 0.5, 0.0]])
    actual_similarity = eval_lib.cosine_similarity(
        orthog_embedding0, orthog_embedding1
    )
    expected_similarity = 0.0
    np.testing.assert_allclose(actual_similarity, expected_similarity)

    opposite_embedding0 = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])
    opposite_embedding1 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    actual_similarity = eval_lib.cosine_similarity(
        opposite_embedding0, opposite_embedding1
    )
    expected_similarity = -1.0
    np.testing.assert_allclose(actual_similarity, expected_similarity)


class TaxonomyModelCallbackTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.workdir = tempfile.TemporaryDirectory('workdir').name

  def test_learned_representation_blocklist(self):
    workdir = os.path.join(self.workdir, '1')
    init_config = config_utils.parse_config(
        baseline_mel_conformer.get_config(), config_globals.get_globals()
    ).init_config
    model_bundle, train_state = classifier.initialize_model(
        workdir=workdir, **init_config
    )
    _ = model_bundle.ckpt.restore_or_initialize(train_state)

    db = namespace_db.load_db()
    all_species = db.class_lists['xenocanto'].classes
    downstream_species = db.class_lists['downstream_species_v2'].classes

    # The model callback should load all available learned representations when
    # use_learned_representations is set to True (by default, set to False).
    self.assertLen(
        callbacks.TaxonomyModelCallback(
            init_config=init_config,
            workdir=workdir,
            use_learned_representations=True,
        ).learned_representations,
        len(all_species),
    )
    # When learned_representation_blocklist is passed, the model callback
    # should *not* load any learned representation for species in the blocklist.
    self.assertNoCommonElements(
        callbacks.TaxonomyModelCallback(
            init_config=init_config,
            workdir=workdir,
            use_learned_representations=True,
            learned_representation_blocklist=downstream_species,
        ).learned_representations.keys(),
        downstream_species,
    )

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.workdir)


if __name__ == '__main__':
  absltest.main()
