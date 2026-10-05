# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for undersubscribed call-density sample bins."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from chirp.inference import call_density
from chirp.inference.search import search
import numpy as np


class CallDensitySamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    state = np.random.get_state()
    self.addCleanup(np.random.set_state, state)
    np.random.seed(17)

  def make_results(self, counts):
    results = []
    for bin_id, count in enumerate(counts):
      for i in range(count):
        results.append(
            search.SearchResult(
                embedding=np.array([bin_id, i], dtype=np.float32),
                score=bin_id * 10 + 1 + i / (count + 1),
                sort_score=float(len(results)),
                filename=f'{bin_id}_{i}.wav',
                timestamp_offset=0.0,
            )
        )
    return search.TopKSearchResults(len(results), results)

  @parameterized.product(
      counts=[
          (0, 0, 0),
          (1, 0, 5),
          (3, 1, 2),
          (2, 2, 2),
          (5, 5, 5),
          (1, 1, 1),
          (0, 4, 4),
          (5, 0, 0),
      ],
      limit=[0, 2],
  )
  def test_caps_each_bin_at_available_results(self, counts, limit):
    results = self.make_results(counts)
    original_order = [id(r) for r in results.search_results]
    expected_counts = [min(c, limit) for c in counts]
    output = call_density.prune_random_results(
        results,
        np.array([0.0, 10.0, 20.0, 30.0]),
        np.array([0.0, 1 / 3, 2 / 3, 1.0]),
        limit,
    )

    self.assertLen(output.search_results, sum(expected_counts))
    self.assertEqual(output.top_k, sum(expected_counts))
    selected = collections.Counter(
        int(r.filename.split('_')[0]) for r in output.search_results
    )
    self.assertEqual([selected[i] for i in range(3)], expected_counts)
    self.assertLen({id(r) for r in output.search_results}, sum(expected_counts))
    self.assertTrue({id(r) for r in output.search_results}.issubset(original_order))
    self.assertEqual([id(r) for r in results.search_results], original_order)
    self.assertEqual(
        [r.sort_score for r in output],
        sorted([r.sort_score for r in output.search_results], reverse=True),
    )

  def test_well_populated_bins_keep_seeded_choices(self):
    results = self.make_results((5, 5, 5))
    bins = [
        [r for r in results.search_results if r.filename.startswith(f'{i}_')]
        for i in range(3)
    ]
    seed = np.random.get_state()
    expected = []
    for bucket in bins:
      expected.extend(np.random.choice(bucket, 2, replace=False))
    np.random.default_rng(42).shuffle(expected)
    expected = search.TopKSearchResults(len(expected), expected)
    expected_state = np.random.get_state()
    np.random.set_state(seed)

    actual = call_density.prune_random_results(
        results,
        np.array([0.0, 10.0, 20.0, 30.0]),
        np.array([0.0, 1 / 3, 2 / 3, 1.0]),
        2,
    )
    self.assertEqual([r.filename for r in actual], [r.filename for r in expected])
    actual_state = np.random.get_state()
    self.assertEqual(actual_state[0], expected_state[0])
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    self.assertEqual(actual_state[2:], expected_state[2:])

  def test_repeated_quantile_bounds_allow_empty_bins(self):
    results = self.make_results((1, 0, 0))
    actual = call_density.prune_random_results(
        results, np.ones(5), np.linspace(0, 1, 5), 3
    )
    self.assertLen(actual.search_results, 1)
    self.assertIs(actual.search_results[0], results.search_results[0])

  def test_negative_sample_request_still_raises(self):
    with self.assertRaises(ValueError):
      call_density.prune_random_results(
          self.make_results((3, 3, 3)),
          np.array([0.0, 10.0, 20.0, 30.0]),
          np.array([0.0, 1 / 3, 2 / 3, 1.0]),
          -1,
      )


if __name__ == '__main__':
  absltest.main()
