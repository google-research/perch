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

"""Tests for Hoplite indexing."""

import shutil
import tempfile

from chirp.projects.hoplite import index
from chirp.projects.hoplite.tests import test_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

EMBEDDING_SIZE = 8


class IndexTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_index_delegates(self):
    num_embeddings = 100
    rng = np.random.default_rng(42)
    db = test_utils.make_db(
        self.tempdir, 'in_mem', num_embeddings, rng, EMBEDDING_SIZE
    )
    v = index.HopliteSearchIndex.from_db(db, score_fn_name='dot')
    roots = v.index_delegates(degree_bound=3, num_tree_iterations=3)
    # The first root is usually db.get_one_embedding_id(), so query some other
    # embedding.
    ids = db.get_embedding_ids()
    rng.shuffle(ids)
    q = ids[0]
    q_emb = db.get_embedding(q)
    with self.subTest('greedy_search'):
      results, _ = v.greedy_search(q_emb, roots[0], search_list_size=10)
      result_ids = [r.embedding_id for r in results]
      self.assertIn(q, result_ids)


if __name__ == '__main__':
  absltest.main()
