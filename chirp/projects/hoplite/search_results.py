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

"""Search results containers."""

import dataclasses
import heapq


@dataclasses.dataclass
class SearchResult:
  """Container for a single search result."""

  # Embedding ID.
  embedding_id: int
  # Score used for sorting the result.
  sort_score: float

  def __lt__(self, other):
    return self.sort_score < other.sort_score

  def __gt__(self, other):
    return self.sort_score > other.sort_score

  def __le__(self, other):
    return self.sort_score <= other.sort_score

  def __ge__(self, other):
    return self.sort_score >= other.sort_score


@dataclasses.dataclass
class TopKSearchResults:
  """Top-K search results."""

  top_k: int
  search_results: list[SearchResult] = dataclasses.field(default_factory=list)
  _ids: set[int] = dataclasses.field(default_factory=set)

  def __post_init__(self):
    heapq.heapify(self.search_results)
    self._ids = set(q.embedding_id for q in self.search_results)

  def __iter__(self):
    for q in sorted(self.search_results, reverse=True):
      yield q

  def update(self, search_result: SearchResult, force_insert=False):
    """Update Results with the new result."""
    if not force_insert and self.will_filter(
        search_result.embedding_id, search_result.sort_score
    ):
      return
    if len(self.search_results) >= self.top_k:
      popped = heapq.heappop(self.search_results).embedding_id
      self._ids.remove(popped)
    heapq.heappush(self.search_results, search_result)
    self._ids.add(search_result.embedding_id)

  @property
  def min_score(self) -> float:
    return self.search_results[0].sort_score

  def will_filter(self, idx: int, score: float) -> bool:
    """Check whether a score is relevant."""
    if idx in self._ids:
      return True
    if len(self.search_results) < self.top_k:
      # Add the result, regardless of score, until we have k results.
      return False
    return score < self.min_score
