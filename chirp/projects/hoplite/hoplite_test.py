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

from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import index
from chirp.projects.hoplite import sqlite_impl
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class HopliteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def _make_db(self, db_type, num_embeddings: int, rng: np.random.Generator):
    if db_type == 'in_mem':
      db = in_mem_impl.InMemoryGraphSearchDB.create()

    elif db_type == 'sqlite':
      # TODO(tomdenton): use tempfile.
      db = sqlite_impl.SQLiteGraphSearchDB.create(
          db_path=os.path.join(self.tempdir, 'db.sqlite')
      )
    else:
      raise ValueError(f'Unknown db type: {db_type}')
    db.setup()
    # Insert a few embeddings...
    random_embeddings = np.float16(
        rng.normal(size=[num_embeddings, 128], loc=0, scale=1.0)
    )
    for emb in random_embeddings:
      db.insert_embedding(emb)
    return db

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_db_interface(self, db_type):
    rng = np.random.default_rng(42)
    db = self._make_db(db_type, 1000, rng)

    # Run all methods in the interface...
    self.assertEqual(db.count_embeddings(), 1000)
    self.assertEqual(db.count_edges(), 0)

    # Insert a few random edges...
    idxes = list(range(1, db.count_embeddings() + 1))
    got_idxes = db.get_embedding_ids()
    self.assertSameElements(sorted(got_idxes), idxes)

    rng.shuffle(idxes)
    for i in range(db.count_embeddings() - 1):
      db.insert_edge(idxes[i], idxes[i + 1])
    db.insert_edge(idxes[-1], idxes[0])
    self.assertEqual(db.count_edges(), 1000)

    # Delete a few edges...
    for i in range(0, db.count_embeddings(), 2):
      db.delete_edge(idxes[i], idxes[i + 1])
    self.assertEqual(db.count_edges(), 500)

    print(idxes[1], '->', [idxes[3], idxes[5], idxes[7]])
    db.insert_edge(idxes[1], idxes[3])
    db.insert_edge(idxes[1], idxes[5])
    db.insert_edge(idxes[1], idxes[7])
    edges = db.get_edges(idxes[1])
    print(edges)
    self.assertSameElements(edges, [idxes[2], idxes[3], idxes[5], idxes[7]])
    self.assertEqual(db.count_edges(), 503)
    db.delete_edges(idxes[1])
    edges = db.get_edges(idxes[1])
    print(edges)
    self.assertEqual(db.count_edges(), 499)

    db.drop_all_edges()
    self.assertEqual(db.count_edges(), 0)

    db.commit()

  def test_impl_agreement(self):
    """Check that in-memory and sqlite operations give the same results."""
    rng = np.random.default_rng(42)
    in_mem_db = self._make_db('in_mem', 1000, rng)
    rng = np.random.default_rng(42)
    sqlite_db = self._make_db('sqlite', 1000, rng)

    in_mem_ids = in_mem_db.get_embedding_ids()
    sqlite_ids = sqlite_db.get_embedding_ids()
    self.assertSameElements(in_mem_ids, sqlite_ids)

    for id_ in in_mem_ids:
      self.assertSameElements(
          in_mem_db.get_embedding(id_), sqlite_db.get_embedding(id_)
      )

    got_m = in_mem_db.get_embeddings((2, 4, 8, 16))
    got_s = sqlite_db.get_embeddings((2, 4, 8, 16))
    self.assertEqual(np.sum(np.square(got_m - got_s)), 0.0)

  def test_brute_search_impl_agreement(self):
    rng = np.random.default_rng(42)
    in_mem_db = self._make_db('in_mem', 1000, rng)
    rng = np.random.default_rng(42)
    sqlite_db = self._make_db('sqlite', 1000, rng)

    # Check brute-force search agreement.
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    results_m = graph_utils.brute_search(
        in_mem_db, query, search_list_size=10, score_fn=np.dot
    )
    results_s = graph_utils.brute_search(
        sqlite_db, query, search_list_size=10, score_fn=np.dot
    )
    self.assertSequenceEqual(
        tuple(r.embedding_id for r in results_m),
        tuple(r.embedding_id for r in results_s),
    )

  def test_greedy_search_impl_agreement(self):
    rng = np.random.default_rng(42)
    in_mem_db = self._make_db('in_mem', 1000, rng)
    rng = np.random.default_rng(42)
    sqlite_db = self._make_db('sqlite', 1000, rng)
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    v_s = index.HopliteSearchIndex.from_db(sqlite_db)
    v_m = index.HopliteSearchIndex.from_db(in_mem_db)
    results_s, path_s = v_s.greedy_search(
        query, search_list_size=32, start_node=1
    )
    results_m, path_m = v_m.greedy_search(
        query, search_list_size=32, start_node=1
    )
    self.assertSequenceEqual(path_s, path_m)
    self.assertSequenceEqual(results_s._ids, results_m._ids)


if __name__ == '__main__':
  absltest.main()
