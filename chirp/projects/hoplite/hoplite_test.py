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

"""Tests for Hoplite."""

import os
import shutil
import tempfile

from chirp.projects.hoplite import db_loader
from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import index
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from ml_collections import config_dict
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

EMBEDDING_SIZE = 128


class HopliteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def _make_db(
      self, db_type: str, num_embeddings: int, rng: np.random.Generator
  ) -> interface.GraphSearchDBInterface:
    if db_type == 'in_mem':
      db = in_mem_impl.InMemoryGraphSearchDB.create(
          embedding_dim=EMBEDDING_SIZE,
          max_size=2000,
          degree_bound=1000,
      )
    elif db_type == 'sqlite':
      # TODO(tomdenton): use tempfile.
      db = sqlite_impl.SQLiteGraphSearchDB.create(
          db_path=os.path.join(self.tempdir, 'other_db.sqlite'),
          embedding_dim=EMBEDDING_SIZE,
      )
    else:
      raise ValueError(f'Unknown db type: {db_type}')
    db.setup()
    # Insert a few embeddings...
    graph_utils.insert_random_embeddings(
        db, EMBEDDING_SIZE, num_embeddings, rng
    )

    config = config_dict.ConfigDict()
    config.embedding_dim = EMBEDDING_SIZE
    db.insert_metadata('db_config', config)
    return db

  def _clone_embeddings(
      self,
      source_db: interface.GraphSearchDBInterface,
      target_db: interface.GraphSearchDBInterface,
  ):
    """Copy all embeddings to target_db and provide an id mapping."""
    id_mapping = {}

    for idx in source_db.get_embedding_ids():
      emb = source_db.get_embedding(idx)
      source = source_db.get_embedding_source(idx)
      target_id = target_db.insert_embedding(emb, source)
      id_mapping[idx] = target_id
    return id_mapping

  def _add_random_edges(
      self,
      rng: np.random.Generator,
      db: interface.GraphSearchDBInterface,
      degree: int,
  ):
    ids = db.get_embedding_ids()
    for idx in ids:
      ys = rng.choice(np.setdiff1d(ids, [idx]), degree, replace=False)
      db.insert_edges(idx, ys)

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_graph_db_interface(self, db_type):
    rng = np.random.default_rng(42)
    db = self._make_db(db_type, 1000, rng)

    # Run all methods in the interface...
    self.assertEqual(db.count_embeddings(), 1000)
    self.assertEqual(db.count_edges(), 0)

    # Insert a few random edges...
    idxes = db.get_embedding_ids()
    self.assertLen(idxes, 1000)

    # Check the metadata.
    got_md = db.get_metadata('db_config')
    self.assertEqual(got_md.embedding_dim, EMBEDDING_SIZE)

    rng.shuffle(idxes)
    for i in range(db.count_embeddings() - 1):
      db.insert_edge(idxes[i], idxes[i + 1])
    db.insert_edge(idxes[-1], idxes[0])
    self.assertEqual(db.count_edges(), 1000)

    # Delete a few edges...
    for i in range(0, db.count_embeddings(), 2):
      db.delete_edge(idxes[i], idxes[i + 1])
    self.assertEqual(db.count_edges(), 500)

    db.insert_edges(idxes[1], np.array([idxes[3], idxes[5], idxes[7]]))
    edges = db.get_edges(idxes[1])
    self.assertSameElements(edges, [idxes[2], idxes[3], idxes[5], idxes[7]])
    self.assertEqual(db.count_edges(), 503)
    db.delete_edges(idxes[1])
    edges = db.get_edges(idxes[1])
    self.assertEmpty(edges)
    self.assertEqual(db.count_edges(), 499)

    db.drop_all_edges()
    self.assertEqual(db.count_edges(), 0)

    db.commit()

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_get_embeddings_by_source(self, db_type):
    rng = np.random.default_rng(42)
    db = self._make_db(db_type, 1000, rng)

    test_id = db.get_one_embedding_id()
    test_source = db.get_embedding_source(test_id)
    with self.subTest('get_embeddings_by_source_dataset'):
      ds_sources = db.get_embeddings_by_source(
          test_source.dataset_name, None, None
      )
      self.assertIn(test_id, ds_sources)
      # About one-third of the embeddings should be from the same dataset.
      self.assertAlmostEqual(
          db.count_embeddings() / 3, ds_sources.shape[0], delta=100
      )

    with self.subTest('get_embeddings_by_source_file_id'):
      ds_sources = db.get_embeddings_by_source(
          test_source.dataset_name, test_source.source_id, None
      )
      self.assertIn(test_id, ds_sources)

    with self.subTest('get_embeddings_by_source_file_id_and_offsets'):
      ds_sources = db.get_embeddings_by_source(
          test_source.dataset_name, test_source.source_id, test_source.offsets
      )
      self.assertIn(test_id, ds_sources)

    with self.subTest('get_embeddings_by_source_miss'):
      ds_sources = db.get_embeddings_by_source(
          'missing_dataset', test_source.source_id, test_source.offsets
      )
      self.assertEqual(0, ds_sources.shape[0])

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_labels_db_interface(self, db_type):
    rng = np.random.default_rng(42)
    db = self._make_db(db_type, 1000, rng)
    ids = db.get_embedding_ids()
    db.insert_label(
        interface.Label(ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'human')
    )
    db.insert_label(
        interface.Label(
            ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'machine'
        )
    )
    db.insert_label(
        interface.Label(
            ids[1], 'hawgoo', interface.LabelType.POSITIVE, 'machine'
        )
    )
    db.insert_label(
        interface.Label(
            ids[0], 'rewbla', interface.LabelType.NEGATIVE, 'machine'
        )
    )

    with self.subTest('get_embeddings_by_label'):
      # When both label_type and source are unspecified, we should get all
      # unique IDs with the target label. Id's 0 and 1 both have some kind of
      # 'hawgoo' label.
      got = db.get_embeddings_by_label('hawgoo', None, None)
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

    with self.subTest('get_embeddings_by_label_type'):
      # Now we should get the ID's for all POSITIVE 'hawgoo' labels, regardless
      # of source.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.POSITIVE, None
      )
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

      # There are no negative 'hawgoo' labels.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.NEGATIVE, None
      )
      self.assertEqual(got.shape[0], 0)

    with self.subTest('get_embeddings_by_label_source'):
      # There is only one hawgoo labeled by a human.
      got = db.get_embeddings_by_label('hawgoo', None, 'human')
      self.assertSequenceEqual(got, [ids[0]])

      # And only one example with a 'rewbla' labeled by a machine.
      got = db.get_embeddings_by_label('rewbla', None, 'machine')
      self.assertSequenceEqual(got, [ids[0]])

    with self.subTest('count_all_labels'):
      # Finally, there are a total of three labels on ID 0.
      got = db.get_labels(ids[0])
      self.assertLen(got, 3)

  def test_brute_search_impl_agreement(self):
    rng = np.random.default_rng(42)
    in_mem_db = self._make_db('in_mem', 1000, rng)

    sqlite_db = self._make_db('sqlite', 0, rng)
    id_mapping = self._clone_embeddings(in_mem_db, sqlite_db)

    # Check brute-force search agreement.
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    results_m, _ = graph_utils.brute_search(
        in_mem_db, query, search_list_size=10, score_fn=np.dot
    )
    results_s, _ = graph_utils.brute_search(
        sqlite_db, query, search_list_size=10, score_fn=np.dot
    )
    self.assertLen(results_m.search_results, 10)
    self.assertLen(results_s.search_results, 10)
    # Search results are iterated over in sorted order.
    for r_m, r_s in zip(results_m, results_s):
      emb_m = in_mem_db.get_embedding(r_m.embedding_id)
      emb_s = sqlite_db.get_embedding(r_s.embedding_id)
      self.assertEqual(id_mapping[r_m.embedding_id], r_s.embedding_id)
      # TODO(tomdenton): check that the scores are the same.
      np.testing.assert_equal(emb_m, emb_s)

  def test_greedy_search_impl_agreement(self):
    rng = np.random.default_rng(42)
    in_mem_db = self._make_db('in_mem', 1000, rng)
    self._add_random_edges(rng, in_mem_db, degree=10)

    sqlite_db = self._make_db('sqlite', 0, rng)
    id_mapping = self._clone_embeddings(in_mem_db, sqlite_db)

    for x in in_mem_db.get_embedding_ids():
      nbrs = in_mem_db.get_edges(x)
      for y in nbrs:
        sqlite_db.insert_edge(id_mapping[x], id_mapping[y])

    rng = np.random.default_rng(42)
    query = rng.normal(size=(EMBEDDING_SIZE,), loc=0, scale=1.0)

    v_m = index.HopliteSearchIndex.from_db(in_mem_db)
    v_s = index.HopliteSearchIndex.from_db(sqlite_db)

    results_m, path_m = v_m.greedy_search(
        query, search_list_size=32, start_node=0, deterministic=True
    )
    results_s, path_s = v_s.greedy_search(
        query, search_list_size=32, start_node=1, deterministic=True
    )
    self.assertSameElements((id_mapping[x] for x in path_m), path_s)

    # Iterating over the search results proceeds in sorted order.
    np.testing.assert_equal(
        [id_mapping[r.embedding_id] for r in results_m.search_results],
        [r.embedding_id for r in results_s.search_results],
    )

  @parameterized.product(
      source_db_type=(
          'in_mem',
          'sqlite',
      ),
      target_db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_duplicate_db(self, source_db_type, target_db_type):
    rng = np.random.default_rng(42)
    source_db = self._make_db(source_db_type, 1000, rng)
    self._add_random_edges(rng, source_db, degree=10)

    target_db_config = config_dict.ConfigDict()
    target_db_config.embedding_dim = source_db.embedding_dimension()
    if target_db_type == 'sqlite':
      target_db_config.db_path = os.path.join(self.tempdir, 'db.sqlite')
    elif target_db_type == 'in_mem':
      target_db_config.max_size = 2000
      target_db_config.degree_bound = 1000
    else:
      raise ValueError(f'Unknown target_db_type: {target_db_type}')

    labeled_idx = source_db.get_one_embedding_id()
    source_db_label = interface.Label(
        labeled_idx, 'hawgoo', interface.LabelType.POSITIVE, 'human'
    )
    source_db.insert_label(source_db_label)

    target_db, id_mapping = db_loader.duplicate_db(
        source_db, target_db_type, target_db_config
    )
    self.assertLen(id_mapping, 1000)

    with self.subTest('embeddings'):
      # Check that the target DB is a faithful copy of the source DB.
      for idx in source_db.get_embedding_ids():
        source_emb = source_db.get_embedding(idx)
        source_source = source_db.get_embedding_source(idx)
        target_emb = target_db.get_embedding(id_mapping[idx])
        target_source = target_db.get_embedding_source(id_mapping[idx])
        np.testing.assert_array_equal(source_emb, target_emb)
        self.assertEqual(target_source.dataset_name, source_source.dataset_name)
        self.assertEqual(target_source.source_id, source_source.source_id)
        np.testing.assert_array_equal(
            source_source.offsets, target_source.offsets
        )

    with self.subTest('edges'):
      for idx in source_db.get_embedding_ids():
        # Check that the edges are the same.
        source_nbrs = source_db.get_edges(idx)
        target_nbrs = target_db.get_edges(id_mapping[idx])
        mapped_source_nbrs = np.array([id_mapping[nbr] for nbr in source_nbrs])
        self.assertSameElements(mapped_source_nbrs, target_nbrs)

    with self.subTest('metadata'):
      source_metadata = source_db.get_metadata(key=None)
      target_metadata = target_db.get_metadata(key=None)
      for k, v in source_metadata.items():
        if k == 'db_config':
          continue
        self.assertEqual(v, target_metadata[k])

    with self.subTest('labels'):
      source_labels = source_db.get_labels(labeled_idx)
      target_labels = target_db.get_labels(id_mapping[labeled_idx])
      self.assertLen(source_labels, 1)
      self.assertLen(target_labels, 1)
      self.assertEqual(source_labels[0].label, target_labels[0].label)
      self.assertEqual(source_labels[0].type, target_labels[0].type)
      self.assertEqual(source_labels[0].provenance, target_labels[0].provenance)
      self.assertEqual(
          id_mapping[source_labels[0].embedding_id],
          target_labels[0].embedding_id,
      )


if __name__ == '__main__':
  absltest.main()
