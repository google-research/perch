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

"""Tests for brute search functionality."""

import shutil
import tempfile

from chirp.projects.hoplite import brutalism
from chirp.projects.hoplite.tests import test_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

EMBEDDING_SIZE = 8


class BrutalismTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite',
      ),
  )
  def test_threaded_brute_search(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    query_idx = db.get_one_embedding_id()
    query_embedding = db.get_embedding(query_idx)
    results, scores = brutalism.brute_search(
        db,
        query_embedding,
        search_list_size=10,
        score_fn=np.dot,
    )
    self.assertSequenceEqual(scores.shape, (1000,))
    self.assertLen(results.search_results, 10)
    got_ids = [r.embedding_id for r in results]
    self.assertIn(query_idx, got_ids)

    # Check agreement of threaded brute search with the non-threaded version.
    t_results, t_scores = brutalism.threaded_brute_search(
        db,
        query_embedding,
        search_list_size=10,
        batch_size=128,
        score_fn=np.dot,
    )
    np.testing.assert_equal(np.sort(t_scores), np.sort(scores))
    t_got_ids = [r.embedding_id for r in t_results]
    self.assertSequenceEqual(got_ids, t_got_ids)


if __name__ == '__main__':
  absltest.main()
