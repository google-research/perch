# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Tools for searching an embeddings dataset."""

import dataclasses
from typing import Any, Callable, List, Sequence

import numpy as np
import tensorflow as tf
import tqdm


@dataclasses.dataclass
class SearchResult:
  embedding: np.ndarray
  distance: float
  filename: str
  timestamp_offset: int
  # Audio and label_widgets are populated as needed.
  audio: np.ndarray | None = None
  labels_widgets: Sequence[Any] = ()

  def __hash__(self):
    """Return an identifier for this result."""
    return hash((self.filename, self.timestamp_offset))


@dataclasses.dataclass
class TopKSearchResults:
  """Top-K search results."""

  search_results: List[SearchResult]
  top_k: int
  distance_offset: float = 0.0
  max_dist: float = -1.0
  _max_dist_idx: int = -1

  def __iter__(self):
    for r in self.search_results:
      yield r

  def update(self, search_result):
    """Update Results with the new result."""
    if len(self.search_results) < self.top_k:
      # Add the result, regardless of distance, until we have k results.
      pass
    elif search_result.distance > self.max_dist:
      # Early return to save compute.
      return
    elif len(self.search_results) >= self.top_k:
      self.search_results.pop(self._max_dist_idx)
    self.search_results.append(search_result)
    self._update_deseridata()

  def _update_deseridata(self):
    self._max_dist_idx = np.argmax([r.distance for r in self.search_results])
    self.max_dist = self.search_results[self._max_dist_idx].distance

  def sort(self):
    """Sort the results."""
    distances = np.array([r.distance for r in self.search_results])
    idxs = np.argsort(distances)
    self.search_results = [self.search_results[idx] for idx in idxs]
    self._update_deseridata()


@dataclasses.dataclass
class DistanceStats:
  min_dist: float
  max_dist: float
  mean_dist: float
  std_dist: float
  num_windows: int


def search_embeddings_parallel(
    embeddings_dataset: tf.data.Dataset,
    query_embedding_batch: np.ndarray,
    hop_size: int,
    top_k: int = 10,
    target_dist: float = 0.0,
    query_reduce_fn: Callable = tf.reduce_min,  # pylint: disable=g-bare-generic
):
  """Run a brute-force search.

  Uses tf dataset manipulation to parallelize.

  Args:
    embeddings_dataset: tf.data.Dataset over embeddings
    query_embedding_batch: Batch of query embeddings with shape [Batch, Depth].
    hop_size: Embedding hop size in samples.
    top_k: Number of results desired.
    target_dist: Get results closest to the target_dist. Set to 0.0 for standard
      nearest-neighbor search.
    query_reduce_fn: Tensorflow op for reducing embedding distances to queries.
      One of tf.reduce_min or tf.reduce_mean is probably what you want.

  Returns:
    TopKSearchResults and distance statistics reduced per-file.
  """
  # Expand from shape [B, D] to shape [B, T', C', D]
  queries = query_embedding_batch[:, np.newaxis, np.newaxis, :]

  def _q_dist(ex):
    # exapand embedding from [T, C, D] to [B', T, C, D].
    dists = (ex['embedding'][tf.newaxis, :, :, :] - queries) ** 2
    # Take min distance over channels and queries, leaving only time.
    dists = tf.reduce_sum(dists, axis=-1)  # Reduce over vector depth
    dists = tf.reduce_min(dists, axis=-1)  # Reduce over channels
    dists = tf.math.sqrt(dists)
    dists = query_reduce_fn(dists, axis=0)  # Reduce over query batch

    ex['q_distance'] = dists
    ex['min_dist'] = tf.reduce_min(dists)
    ex['max_dist'] = tf.reduce_max(dists)
    ex['mean_dist'] = tf.reduce_mean(dists)
    ex['std_dist'] = tf.math.reduce_std(dists)
    return ex

  embeddings_dataset = embeddings_dataset.map(
      _q_dist, num_parallel_calls=tf.data.AUTOTUNE
  )

  results = TopKSearchResults([], top_k=top_k, distance_offset=target_dist)
  file_stats = {}
  for ex in tqdm.tqdm(embeddings_dataset.as_numpy_iterator()):
    for t in range(ex['embedding'].shape[0]):
      dist = np.abs(ex['q_distance'][0, t] - target_dist)
      offset = t * hop_size + ex['timestamp_offset']
      result = SearchResult(
          ex['embedding'][t, :, :], dist, ex['filename'].decode(), offset
      )
      results.update(result)
    file_stats[ex['filename'].decode()] = DistanceStats(
        min_dist=ex['min_dist'],
        max_dist=ex['max_dist'],
        mean_dist=ex['mean_dist'],
        std_dist=ex['std_dist'],
        num_windows=ex['embedding'].shape[0],
    )
  results.sort()
  return results, file_stats
