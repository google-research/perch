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

"""Tests for jax-based indexing functionality."""

import functools

from chirp.projects.hoplite import index
from chirp.projects.hoplite import index_jax
from chirp.projects.hoplite.tests import test_utils
from jax import numpy as jnp
import numpy as np

from absl.testing import absltest


class IndexJaxTest(absltest.TestCase):

  def test_unique1d(self):
    with self.subTest('rank_one'):
      v = jnp.array([1, 2, 3, 1, 4, 2, 5])
      unique = index_jax.unique1d(v)
      expected = np.array([1, 2, 3, -1, 4, -1, 5])
      np.testing.assert_array_equal(unique, expected)

    with self.subTest('rank_two'):
      v = jnp.array([[1, 2, 3, 1, 4, 2, 5], [33, 44, 55, 55, 66, 77, 66]])
      unique = index_jax.unique1d(v)
      expected = np.array(
          [[1, 2, 3, -1, 4, -1, 5], [33, 44, 55, -1, 66, 77, -1]]
      )
      np.testing.assert_array_equal(unique, expected)

  def test_cosort(self):
    scores = jnp.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    values = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_scores = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_values = jnp.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    with self.subTest('two_arrays'):
      sorted_scores, sorted_values = index_jax.cosort(scores, values)
      np.testing.assert_array_equal(sorted_scores, expected_scores)
      np.testing.assert_array_equal(sorted_values, expected_values)

    with self.subTest('three_arrays'):
      other_values = jnp.array(
          [1.1, 3.1, 5.1, 7.1, 9.1, 2.1, 4.1, 6.1, 8.1, 10.1]
      )
      expected_other_values = jnp.array(
          [10.1, 8.1, 6.1, 4.1, 2.1, 9.1, 7.1, 5.1, 3.1, 1.1]
      )
      sorted_scores, sorted_values, sorted_other_values = index_jax.cosort(
          scores, values, other_values
      )
      np.testing.assert_array_equal(sorted_scores, expected_scores)
      np.testing.assert_array_equal(sorted_values, expected_values)
      np.testing.assert_array_equal(sorted_other_values, expected_other_values)

  def test_update_delegates(self):
    n = 16
    embedding_dim = 32
    degree_bound = 2
    np.random.seed(42)
    embs = jnp.float16(np.random.normal(size=[n, embedding_dim]))
    d_lists, d_scores = index_jax.make_delegate_lists(n, degree_bound)

    m = 8
    candidates = jnp.arange(m)
    embs_c = embs[candidates]
    scores_c_c = jnp.tensordot(embs_c, embs_c, axes=(-1, -1))
    new_delegates, new_scores = index_jax.update_delegates(
        d_lists, d_scores, candidates, scores_c_c
    )
    # Check shapes.
    np.testing.assert_array_equal(new_delegates.shape, [m, degree_bound])
    np.testing.assert_array_equal(new_scores.shape, [m, degree_bound])
    # There should be no -1 delegates, since we have enough candidates.
    self.assertEqual(np.sum(new_delegates >= 0), m * degree_bound)
    for c in candidates:
      for i, d in enumerate(new_delegates[c]):
        score = jnp.float16(jnp.dot(embs[c], embs[d]))
        self.assertEqual(score, new_scores[c, i])
    d_lists = d_lists.at[candidates].set(new_delegates)
    d_scores = d_scores.at[candidates].set(new_scores)

    # Now update with the full set of candidates.
    candidates = jnp.arange(n)
    scores_c_c = jnp.tensordot(embs, embs, axes=(-1, -1))
    new_delegates, _ = index_jax.update_delegates(
        d_lists, d_scores, candidates, scores_c_c
    )
    # The new_delegates should contain the top two scores for each embedding,
    # excluding the diagonal.
    safe_scores = jnp.fill_diagonal(scores_c_c, -jnp.inf, inplace=False)
    for c in range(n):
      top_idxes = jnp.argsort(-safe_scores[c])[:2]
      self.assertEqual(new_delegates[c, 0], top_idxes[0])
      self.assertEqual(new_delegates[c, 1], top_idxes[1])

  def test_run_e2e(self):
    rng = np.random.default_rng(seed=22)
    db = test_utils.make_db('test_db', 'in_mem', 1024, rng, embedding_dim=16)
    embs = jnp.float16(db.embeddings[:1024])
    edges = jnp.zeros([1024, 8], dtype=jnp.int32)
    max_delegates = 32
    output_data = index_jax.delegate_indexing(
        embs,
        edges,
        None,
        None,
        sample_size=32,
        max_delegates=max_delegates,
        alpha=0.5,
        max_violations=1,
    )
    np.testing.assert_array_equal(
        output_data.delegate_lists.shape, [1024, max_delegates]
    )
    np.testing.assert_array_equal(
        output_data.delegate_scores.shape, [1024, max_delegates]
    )
    db.edges = np.asarray(output_data.edges).copy()

    v = index.HopliteSearchIndex.from_db(db, score_fn_name='dot')
    search_partial_fn = functools.partial(
        v.greedy_search, start_node=0, search_list_size=128
    )
    search_fn = lambda q: search_partial_fn(q)[0]
    recall = v.multi_test_recall(search_fn)
    self.assertGreater(recall, 0.9)


if __name__ == '__main__':
  absltest.main()
