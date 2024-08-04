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

import shutil
import tempfile

from chirp.projects.hoplite import brutalism
from chirp.projects.hoplite import index
from chirp.projects.hoplite import interface
from chirp.projects.hoplite.tests import test_utils
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
      thread_split=(True, False),
  )
  def test_graph_db_interface(self, db_type, thread_split):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    if thread_split:
      db = db.thread_split()

    # Run all methods in the interface...
    self.assertEqual(db.count_embeddings(), 1000)
    self.assertEqual(db.count_edges(), 0)

    # Insert a few random edges...
    idxes = db.get_embedding_ids()
    self.assertLen(idxes, 1000)

    one_idx = db.get_one_embedding_id()
    self.assertIn(one_idx, idxes)

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
  def test_labels_db_interface(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
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

    with self.subTest('get_classes'):
      got = db.get_classes()
      self.assertSequenceEqual(got, ['hawgoo', 'rewbla'])

    with self.subTest('get_class_counts'):
      # 2 positive labels for 'hawgoo' ignoring provenance, 0 for 'rewbla'.
      got = db.get_class_counts(interface.LabelType.POSITIVE)
      self.assertDictEqual(got, {'hawgoo': 2, 'rewbla': 0})

      # 1 negative label for 'rewbla', 0 for 'hawgoo'.
      got = db.get_class_counts(interface.LabelType.NEGATIVE)
      self.assertDictEqual(got, {'hawgoo': 0, 'rewbla': 1})

    with self.subTest('count_classes'):
      self.assertEqual(db.count_classes(), 2)

  def test_brute_search_impl_agreement(self):
    rng = np.random.default_rng(42)
    in_mem_db = test_utils.make_db(
        self.tempdir, 'in_mem', 1000, rng, EMBEDDING_SIZE
    )
    sqlite_db = test_utils.make_db(
        self.tempdir, 'sqlite', 0, rng, EMBEDDING_SIZE
    )
    id_mapping = test_utils.clone_embeddings(in_mem_db, sqlite_db)

    # Check brute-force search agreement.
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    results_m, _ = brutalism.brute_search(
        in_mem_db, query, search_list_size=10, score_fn=np.dot
    )
    results_s, _ = brutalism.brute_search(
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
    in_mem_db = test_utils.make_db(
        self.tempdir, 'in_mem', 1000, rng, EMBEDDING_SIZE
    )
    self._add_random_edges(rng, in_mem_db, degree=10)

    sqlite_db = test_utils.make_db(
        self.tempdir, 'sqlite', 0, rng, EMBEDDING_SIZE
    )
    id_mapping = test_utils.clone_embeddings(in_mem_db, sqlite_db)

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


if __name__ == '__main__':
  absltest.main()
