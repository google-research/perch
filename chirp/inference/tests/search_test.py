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

"""Tests for the bootstrap search component."""

from chirp.inference.search import search
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class SearchTest(parameterized.TestCase):

  def test_top_k_search_results(self):
    np.random.seed(42)
    random_embeddings = np.random.normal(size=[100, 5])
    query = np.random.normal(size=[1, 5])
    dists = np.sum((random_embeddings - query) ** 2, axis=1)

    fake_results = []
    for i in range(100):
      r = search.SearchResult(
          random_embeddings[i],
          score=dists[i],
          # Sort with negative distance so that the high score is best.
          sort_score=-dists[i],
          filename=f'result_{i:03d}',
          timestamp_offset=i,
      )
      fake_results.append(r)

    results = search.TopKSearchResults(top_k=10)
    for i, r in enumerate(fake_results):
      results.update(r)
      self.assertLen(results.search_results, min([i + 1, 10]))
      # Get the 10th largest value amongst the dists seen so far.
      true_min_neg_dist = -np.max(sorted(dists[: i + 1])[:10])
      arg_min_dist = np.argmin([r.sort_score for r in results.search_results])
      self.assertEqual(results.min_score, true_min_neg_dist)
      self.assertEqual(
          results.search_results[arg_min_dist].sort_score, results.min_score
      )

    self.assertLen(results.search_results, results.top_k)
    last_score = None
    for i, result in enumerate(results):
      if i > 0:
        self.assertGreater(
            last_score,
            result.sort_score,
        )
      last_score = result.sort_score

  @parameterized.product(
      metric_name=('euclidean', 'cosine', 'mip'),
  )
  def test_metric_apis(self, metric_name):
    example = {
        'embedding': np.random.normal(size=[12, 5, 128]),
    }
    query = np.random.normal(size=[3, 128])
    if metric_name == 'euclidean':
      got = search._euclidean_score(example, query)
    elif metric_name == 'cosine':
      got = search._cosine_score(example, query)
    elif metric_name == 'mip':
      got = search._mip_score(example, query)
    else:
      raise ValueError(f'Unknown metric: {metric_name}')
    self.assertIn('scores', got)
    self.assertSequenceEqual(got['scores'].shape, (12,))
    # Embeddings should be unchanged.
    self.assertEqual(np.max(np.abs(got['embedding'] - example['embedding'])), 0)

  def test_update_sort_scores(self):
    example = {
        'embedding': np.random.normal(size=[12, 5, 128]),
        'scores': np.random.normal(size=[12]),
    }
    got = search._update_sort_scores(example, invert=False, target_score=None)
    self.assertIn('sort_scores', got)
    self.assertSequenceEqual(got['sort_scores'].shape, got['scores'].shape)
    self.assertIn('max_sort_score', got)
    self.assertEqual(np.max(example['scores']), got['max_sort_score'])
    # Embeddings should be unchanged.
    self.assertEqual(np.max(np.abs(got['embedding'] - example['embedding'])), 0)

    got = search._update_sort_scores(example, invert=True, target_score=None)
    self.assertIn('sort_scores', got)
    self.assertSequenceEqual(got['sort_scores'].shape, got['scores'].shape)
    self.assertIn('max_sort_score', got)
    self.assertEqual(-np.min(example['scores']), got['max_sort_score'])
    # Embeddings should be unchanged.
    self.assertEqual(np.max(np.abs(got['embedding'] - example['embedding'])), 0)

    got = search._update_sort_scores(example, invert=False, target_score=1.0)
    self.assertIn('sort_scores', got)
    self.assertSequenceEqual(got['sort_scores'].shape, got['scores'].shape)
    self.assertIn('max_sort_score', got)
    expect_max_score = np.max(1.0 / (np.abs(example['scores'] - 1.0) + 1e-12))
    self.assertEqual(got['max_sort_score'], expect_max_score)
    # Embeddings should be unchanged.
    self.assertEqual(np.max(np.abs(got['embedding'] - example['embedding'])), 0)


if __name__ == '__main__':
  absltest.main()
