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

"""Tests for the bootstrap search component."""

from chirp.projects.bootstrap import search
import numpy as np

from absl.testing import absltest


class SearchTest(absltest.TestCase):

  def test_top_k_search_results(self):
    np.random.seed(42)
    random_embeddings = np.random.normal(size=[100, 5])
    query = np.random.normal(size=[1, 5])
    dists = np.sum((random_embeddings - query) ** 2, axis=1)

    fake_results = []
    for i in range(100):
      r = search.SearchResult(
          random_embeddings[i],
          dists[i],
          filename=f'result_{i:03d}',
          timestamp_offset=i,
      )
      fake_results.append(r)

    results = search.TopKSearchResults([], top_k=10, distance_offset=0.0)
    for i, r in enumerate(fake_results):
      results.update(r)
      self.assertLen(results.search_results, min([i + 1, 10]))
      # Get the 10th largest value amongst the dists seen so far.
      true_max_dist = np.max(sorted(dists[: i + 1])[:10])
      arg_max_dist = np.argmax([r.distance for r in results])
      self.assertEqual(results.max_dist, true_max_dist)
      self.assertEqual(
          results.search_results[arg_max_dist].distance, results.max_dist
      )

    self.assertLen(results.search_results, results.top_k)
    results.sort()
    for i in range(1, 10):
      self.assertGreater(
          results.search_results[i].distance,
          results.search_results[i - 1].distance,
      )


if __name__ == '__main__':
  absltest.main()
