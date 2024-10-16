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

"""Tests for Hoplite databases."""

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
DB_TYPES = ('in_mem', 'sqlite')
DB_TYPE_NAMED_PAIRS = (('in_mem-sqlite', 'in_mem', 'sqlite'),)
PERSISTENT_DB_TYPES = ('sqlite',)


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
      db_type=DB_TYPES,
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

    with self.subTest('test_embedding_sources'):
      source = db.get_embedding_source(idxes[1])
      # The embeddings are given one of three randomly selected dataset names.
      embs = db.get_embeddings_by_source(source.dataset_name, None, None)
      self.assertGreater(embs.shape[0], db.count_embeddings() / 6)
      # For an unknown dataset name, we should get no embeddings.
      embs = db.get_embeddings_by_source('fake_name', None, None)
      self.assertEqual(embs.shape[0], 0)
      # Source ids are approximately unique.
      embs = db.get_embeddings_by_source(
          source.dataset_name, source.source_id, None
      )
      self.assertLen(embs, 1)
      # For an unknown source id, we should get no embeddings.
      embs = db.get_embeddings_by_source(source.dataset_name, 'fake_id', None)
      self.assertEqual(embs.shape[0], 0)

    db.drop_all_edges()
    self.assertEqual(db.count_edges(), 0)

    db.commit()

  @parameterized.product(db_type=PERSISTENT_DB_TYPES)
  def test_persistence(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    self._add_random_edges(rng, db, degree=10)
    one_emb = np.random.normal(size=(EMBEDDING_SIZE,), loc=0, scale=0.05)
    one_emb_id = db.insert_embedding(
        one_emb, source=interface.EmbeddingSource('q', 'x', np.array([5.0]))
    )
    self.assertEqual(db.get_embedding_ids().shape[0], 1001)
    db.commit()

    got_emb = db.get_embedding(one_emb_id)
    np.testing.assert_equal(got_emb, np.float16(one_emb))

    # "Making" the persistent DB without adding any new embeddings gives us a
    # view of the saved DB.
    test_db = test_utils.make_db(self.tempdir, db_type, 0, rng, EMBEDDING_SIZE)
    self.assertIn(one_emb_id, test_db.get_embedding_ids())
    # Check that the embeddings are the same in the two DB's.
    for idx in db.get_embedding_ids():
      emb = db.get_embedding(idx)
      test_emb = test_db.get_embedding(idx)
      np.testing.assert_equal(emb, test_emb)

    for idx in db.get_embedding_ids():
      edges = db.get_edges(idx)
      test_edges = test_db.get_edges(idx)
      self.assertSameElements(edges, test_edges)

  @parameterized.product(db_type=DB_TYPES)
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
      # When both label_type and provenance are unspecified, we should get all
      # unique IDs with the target label. Id's 0 and 1 both have some kind of
      # 'hawgoo' label.
      got = db.get_embeddings_by_label('hawgoo', None, None)
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

    with self.subTest('get_embeddings_by_label_type'):
      # Now we should get the ID's for all POSITIVE 'hawgoo' labels, regardless
      # of provenance.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.POSITIVE, None
      )
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

      # There are no negative 'hawgoo' labels.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.NEGATIVE, None
      )
      self.assertEqual(got.shape[0], 0)

    with self.subTest('get_embeddings_by_label_provenance'):
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

    with self.subTest('duplicate_labels'):
      dupe_label = interface.Label(
          ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'human'
      )
      self.assertFalse(db.insert_label(dupe_label, skip_duplicates=True))
      self.assertTrue(db.insert_label(dupe_label, skip_duplicates=False))

  @parameterized.named_parameters(*DB_TYPE_NAMED_PAIRS)
  def test_brute_search_impl_agreement(self, target_db_type, source_db_type):
    rng = np.random.default_rng(42)
    source_db = test_utils.make_db(
        self.tempdir, source_db_type, 1000, rng, EMBEDDING_SIZE
    )
    target_db = test_utils.make_db(
        self.tempdir, target_db_type, 0, rng, EMBEDDING_SIZE
    )
    id_mapping = test_utils.clone_embeddings(source_db, target_db)

    # Check brute-force search agreement.
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    results_m, _ = brutalism.brute_search(
        source_db, query, search_list_size=10, score_fn=np.dot
    )
    results_s, _ = brutalism.brute_search(
        target_db, query, search_list_size=10, score_fn=np.dot
    )
    self.assertLen(results_m.search_results, 10)
    self.assertLen(results_s.search_results, 10)
    # Search results are iterated over in sorted order.
    for r_m, r_s in zip(results_m, results_s):
      emb_m = source_db.get_embedding(r_m.embedding_id)
      emb_s = target_db.get_embedding(r_s.embedding_id)
      self.assertEqual(id_mapping[r_m.embedding_id], r_s.embedding_id)
      # TODO(tomdenton): check that the scores are the same.
      np.testing.assert_equal(emb_m, emb_s)

  @parameterized.named_parameters(*DB_TYPE_NAMED_PAIRS)
  def test_greedy_search_impl_agreement(self, source_db_type, target_db_type):
    rng = np.random.default_rng(42)
    source_db = test_utils.make_db(
        self.tempdir, source_db_type, 1000, rng, EMBEDDING_SIZE
    )
    self._add_random_edges(rng, source_db, degree=10)
    source_db.commit()

    target_db = test_utils.make_db(
        self.tempdir, target_db_type, 0, rng, EMBEDDING_SIZE
    )
    id_mapping = test_utils.clone_embeddings(source_db, target_db)

    for x in source_db.get_embedding_ids():
      nbrs = source_db.get_edges(x)
      for y in nbrs:
        target_db.insert_edge(id_mapping[x], id_mapping[y])
    target_db.commit()

    rng = np.random.default_rng(42)
    query = rng.normal(size=(EMBEDDING_SIZE,), loc=0, scale=0.05)

    v_s = index.HopliteSearchIndex.from_db(source_db)
    v_t = index.HopliteSearchIndex.from_db(target_db)

    start_node = source_db.get_one_embedding_id()
    results_s, path_s = v_s.greedy_search(
        query, search_list_size=32, start_node=start_node, deterministic=True
    )
    results_t, path_t = v_t.greedy_search(
        query,
        search_list_size=32,
        start_node=id_mapping[start_node],
        deterministic=True,
    )
    self.assertSameElements((id_mapping[x] for x in path_s), path_t)

    # Iterating over the search results proceeds in sorted order.
    np.testing.assert_equal(
        [id_mapping[r.embedding_id] for r in results_s.search_results],
        [r.embedding_id for r in results_t.search_results],
    )


if __name__ == '__main__':
  absltest.main()
