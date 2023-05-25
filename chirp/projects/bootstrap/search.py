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

from chirp.inference import tf_examples
from etils import epath
import numpy as np
from scipy.io import wavfile
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

  def write_labeled_data(self, labeled_data_path: str, sample_rate: int):
    """Write labeled results to the labeled data collection."""
    labeled_data_path = epath.Path(labeled_data_path)
    for r in self.search_results:
      labels = [ch.description for ch in r.label_widgets if ch.value]
      if not labels:
        continue
      extension = epath.Path(r.filename).suffix
      filename = epath.Path(r.filename).name[: -len(extension)]
      output_filename = f'{filename}___{r.timestamp_offset}{extension}'
      for label in labels:
        output_path = labeled_data_path / label
        output_path.mkdir(parents=True, exist_ok=True)
        output_filepath = output_path / output_filename
        wavfile.write(output_filepath, sample_rate, r.audio)


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
    hop_size_s: int,
    top_k: int = 10,
    target_dist: float = 0.0,
    query_reduce_fn: Callable = tf.reduce_min,  # pylint: disable=g-bare-generic
):
  """Run a brute-force search.

  Uses tf dataset manipulation to parallelize.

  Args:
    embeddings_dataset: tf.data.Dataset over embeddings
    query_embedding_batch: Batch of query embeddings with shape [Batch, Depth].
    hop_size_s: Embedding hop size in seconds.
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
    dists = (ex[tf_examples.EMBEDDING][tf.newaxis, :, :, :] - queries) ** 2
    # Take min distance over channels and queries, leaving only time.
    dists = tf.reduce_sum(dists, axis=-1)  # Reduce over vector depth
    dists = tf.reduce_min(dists, axis=-1)  # Reduce over channels
    dists = tf.math.sqrt(dists)
    dists = query_reduce_fn(dists, axis=0)  # Reduce over query batch

    ex['q_distance'] = dists
    return ex

  embeddings_dataset = embeddings_dataset.map(
      _q_dist, num_parallel_calls=tf.data.AUTOTUNE
  )

  results = TopKSearchResults([], top_k=top_k, distance_offset=target_dist)
  all_distances = []
  try:
    for ex in tqdm.tqdm(embeddings_dataset.as_numpy_iterator()):
      all_distances.append(ex['q_distance'].reshape([-1]))
      for t in range(ex[tf_examples.EMBEDDING].shape[0]):
        dist = np.abs(ex['q_distance'][0, t] - target_dist)
        offset = t * hop_size_s + ex[tf_examples.TIMESTAMP_S]
        result = SearchResult(
            ex[tf_examples.EMBEDDING][t, :, :],
            dist,
            ex['filename'].decode(),
            offset,
        )
        results.update(result)
  except KeyboardInterrupt:
    pass
  all_distances = np.concatenate(all_distances)
  results.sort()
  return results, all_distances


def classifer_search_embeddings_parallel(
    embeddings_dataset: tf.data.Dataset,
    embeddings_classifier: tf.keras.Model,
    target_index: int,
    hop_size_s: int,
    top_k: int = 10,
    target_logit: float = 0.0,
):
  """Get examples for a target class with logit near the target logit.

  Args:
    embeddings_dataset: tf.data.Dataset over embeddings
    embeddings_classifier: Keras model turning embeddings into logits.
    target_index: Choice of class index.
    hop_size_s: Embedding hop size in seconds.
    top_k: Number of results desired.
    target_logit: Get results near the target logit.

  Returns:
    TopKSearchResults and all logits.
  """
  results = TopKSearchResults([], top_k=top_k, distance_offset=target_logit)

  def classify_batch(batch):
    emb = batch[tf_examples.EMBEDDING]
    # This seems to 'just work' when the classifier input shape is [None, D]
    # and the embeddings shape is [B, C, D].
    logits = embeddings_classifier(emb)
    # Restrict to target class.
    logits = logits[..., target_index]
    # Take the maximum logit over channels.
    logits = tf.reduce_max(logits, axis=-1)
    batch['logits'] = logits
    return batch

  ds = embeddings_dataset.map(
      classify_batch, num_parallel_calls=tf.data.AUTOTUNE
  )

  all_logits = []
  try:
    for ex in tqdm.tqdm(ds.as_numpy_iterator()):
      emb = ex[tf_examples.EMBEDDING]
      logits = ex['logits']
      all_logits.append(logits)
      for t in range(emb.shape[0]):
        dist = np.abs(logits[t] - target_logit)
        offset = t * hop_size_s + ex[tf_examples.TIMESTAMP_S]
        result = SearchResult(
            emb[t, :, :], dist, ex['filename'].decode(), offset
        )
        results.update(result)
  except KeyboardInterrupt:
    pass
  results.sort()
  all_logits = np.concatenate(all_logits)
  return results, all_logits
