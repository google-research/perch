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

"""Tools for searching an embeddings dataset."""

import collections
import dataclasses
import functools
import heapq
from typing import Any, Callable, List, Sequence

from chirp.inference import interface
from chirp.inference import tf_examples
from etils import epath
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import tqdm


@dataclasses.dataclass
class SearchResult:
  """Container for a single search result."""
  # Embedding vector.
  embedding: np.ndarray
  # Raw score for this result.
  score: float
  # Score used for sorting the result.
  sort_score: float
  # Source file contianing corresponding audio.
  filename: str
  # Time offset for audio.
  timestamp_offset: int

  # The following are populated as needed.
  audio: np.ndarray | None = None
  label_widgets: Sequence[Any] = ()

  def __hash__(self):
    """Return an identifier for this result."""
    return hash((self.filename, self.timestamp_offset))

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
  """Wrapper for sorting and handling TopK search results.

  This class maintains a queue of SearchResult objects, sorted by their
  sort_score. When updated with a new SearchResult, the result is either added
  or ignored appropriately. For speed, the `will_filter` method allows checking
  immediately whether a result with a given score will be discarded. The results
  are kept in heap-order for efficeint updating.

  Iterating over the search results will produce a copy of the results, with
  in-order iteration over results from largest to smallest sort_score.
  """

  top_k: int
  search_results: List[SearchResult] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    heapq.heapify(self.search_results)

  def __iter__(self):
    iter_queue = sorted(self.search_results, reverse=True)
    for result in iter_queue:
      yield result

  @property
  def min_score(self):
    return self.search_results[0].sort_score

  def update(self, search_result: SearchResult) -> None:
    """Update Results with the new result."""
    if self.will_filter(search_result.sort_score):
      return
    if len(self.search_results) >= self.top_k:
      heapq.heappop(self.search_results)
    heapq.heappush(self.search_results, search_result)

  def will_filter(self, score: float) -> bool:
    """Check whether a score is relevant."""
    if len(self.search_results) < self.top_k:
      # Add the result, regardless of score, until we have k results.
      return False
    return score < self.search_results[0].sort_score

  def write_labeled_data(self, labeled_data_path: str, sample_rate: int):
    """Write labeled results to the labeled data collection."""
    labeled_data_path = epath.Path(labeled_data_path)
    counts = collections.defaultdict(int)
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
        output_filepath = epath.Path(output_path / output_filename)
        if output_filepath.exists():
          counts[f'{label} exists'] += 1
          continue
        else:
          counts[label] += 1
        with output_filepath.open('wb') as f:
          wavfile.write(f, sample_rate, np.float32(r.audio))
    for label, count in counts.items():
      print(f'Wrote {count} examples for label {label}')


@dataclasses.dataclass
class DistanceStats:
  min_dist: float
  max_dist: float
  mean_dist: float
  std_dist: float
  num_windows: int


def _euclidean_score(ex, query_embedding_batch):
  """Update example with Euclidean distance scores."""
  # Expand queries from shape [B, D] to shape [B, 1, 1, D]
  queries = query_embedding_batch[:, np.newaxis, np.newaxis, :]
  # Expand embedding from shape [T, C, D] to [1, T, C, D].
  embeddings = ex[tf_examples.EMBEDDING][tf.newaxis, :, :, :]

  dists = (embeddings - queries) ** 2
  # Take min distance over channels and queries, leaving only time.
  dists = tf.reduce_sum(dists, axis=-1)  # Reduce over vector depth
  dists = tf.math.sqrt(dists)
  dists = tf.reduce_min(dists, axis=-1)  # Reduce over channels
  dists = tf.reduce_min(dists, axis=0)  # Reduce over query batch
  ex['scores'] = dists
  return ex


def _mip_score(ex, query_embedding_batch):
  """Update example with MIP distance scores."""
  # embedding shape is [T, C, D].
  # queries have shape [B, D]
  keys = ex[tf_examples.EMBEDDING]
  scores = tf.matmul(keys, query_embedding_batch, transpose_b=True)
  # Product has shape [T, C, B]
  # Take max score over channels and queries, leaving only time.
  scores = tf.reduce_max(scores, axis=-1)  # Reduce over query batch
  scores = tf.reduce_max(scores, axis=-1)  # Reduce over channels
  ex['scores'] = scores
  return ex


def _cosine_score(ex, query_embedding_batch):
  """Update example with MIP distance scores."""
  # embedding shape is [T, C, D].
  # queries have shape [B, D]
  keys = ex[tf_examples.EMBEDDING]
  keys_norm = tf.norm(keys, axis=-1, keepdims=True)
  query_norm = tf.norm(query_embedding_batch, axis=-1, keepdims=True)
  keys = keys / keys_norm
  query = query_embedding_batch / query_norm
  scores = tf.matmul(keys, query, transpose_b=True)

  # Product has shape [T, C, B]
  # Take max score over channels and queries, leaving only time.
  scores = tf.reduce_max(scores, axis=-1)  # Reduce over query batch
  scores = tf.reduce_max(scores, axis=-1)  # Reduce over channels
  ex['scores'] = scores
  return ex


def _update_sort_scores(ex, invert: bool, target_score: float | None):
  """Update example with sort scores."""
  if target_score is not None:
    # We need large values to be good, so we use the inverse distance to the
    # target score as our sorting score.
    ex['sort_scores'] = 1.0 / (tf.abs(ex['scores'] - target_score) + 1e-12)
  elif invert:
    ex['sort_scores'] = -ex['scores']
  else:
    ex['sort_scores'] = ex['scores']
  # Precompute the max score in the example, allowing us to save
  # time by skipping irrelevant examples.
  ex['max_sort_score'] = tf.reduce_max(ex['sort_scores'])
  return ex


def _random_sort_scores(ex):
  ex['sort_scores'] = tf.random.uniform(
      [tf.shape(ex[tf_examples.EMBEDDING])[0]]
  )
  ex['max_sort_score'] = tf.reduce_max(ex['sort_scores'])
  return ex


def search_embeddings_parallel(
    embeddings_dataset: tf.data.Dataset,
    query_embedding_batch: np.ndarray | None,
    hop_size_s: int,
    top_k: int = 10,
    target_score: float | None = None,
    score_fn: Callable[[Any, np.ndarray], Any] | str = 'euclidean',  # pylint: disable=g-bare-generic
    random_sample: bool = False,
    invert_sort_score: bool = False,
    filter_fn: Callable[[Any], bool] | None = None,
):
  """Run a brute-force search.

  Uses tf dataset manipulation to parallelize.

  Args:
    embeddings_dataset: tf.data.Dataset over embeddings
    query_embedding_batch: Batch of query embeddings with shape [Batch, Depth],
      or None if the metric does not require queries.
    hop_size_s: Embedding hop size in seconds.
    top_k: Number of results desired.
    target_score: Get results closest to the target_score.
    score_fn: Scoring function to use.
    random_sample: If True, obtain a uniformly random sample of data.
    invert_sort_score: Set to True if low scores are preferable to high scores.
      Ignored if a string score_fn is given.
    filter_fn: Optional predicate for filtering examples.

  Returns:
    TopKSearchResults and distance statistics reduced per-file.
  """

  # Convert string to score_fn.
  if score_fn == 'euclidean':
    score_fn = _euclidean_score
    invert_sort_score = True
  elif score_fn == 'mip':
    score_fn = _mip_score
    invert_sort_score = False
  elif score_fn == 'cosine':
    score_fn = _cosine_score
    invert_sort_score = False
  elif isinstance(score_fn, str):
    raise ValueError(f'Unknown score_fn: {score_fn}')

  if query_embedding_batch is None:
    pass
  elif len(query_embedding_batch.shape) == 1:
    query_embedding_batch = query_embedding_batch[np.newaxis, :]
  elif len(query_embedding_batch.shape) > 2:
    raise ValueError(
        'query_embedding_batch should be rank 1 or 2, but has shape '
        f'{query_embedding_batch.shape}'
    )

  score_fn = functools.partial(
      score_fn, query_embedding_batch=query_embedding_batch
  )
  if random_sample:
    sort_scores_fn = _random_sort_scores
  else:
    sort_scores_fn = functools.partial(
        _update_sort_scores, target_score=target_score, invert=invert_sort_score
    )

  ex_map_fn = lambda ex: sort_scores_fn(score_fn(ex))
  embeddings_dataset = embeddings_dataset.shuffle(1024).map(
      ex_map_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  if filter_fn is not None:
    embeddings_dataset = embeddings_dataset.filter(filter_fn)
  embeddings_dataset = embeddings_dataset.prefetch(1024)

  results = TopKSearchResults(top_k=top_k)
  all_distances = []
  try:
    for ex in tqdm.tqdm(embeddings_dataset.as_numpy_iterator()):
      all_distances.append(ex['scores'].reshape([-1]))
      if results.will_filter(ex['max_sort_score']):
        continue
      for t in range(ex[tf_examples.EMBEDDING].shape[0]):
        offset_s = t * hop_size_s + ex[tf_examples.TIMESTAMP_S]
        result = SearchResult(
            ex[tf_examples.EMBEDDING][t, :, :],
            ex['scores'][t],
            ex['sort_scores'][t],
            ex['filename'].decode(),
            offset_s,
        )
        results.update(result)
  except KeyboardInterrupt:
    pass
  all_distances = np.concatenate(all_distances)
  return results, all_distances


def classifer_search_embeddings_parallel(
    embeddings_classifier: interface.LogitsOutputHead,
    target_index: int,
    **kwargs,
):
  """Get examples for a target class with logit near the target logit.

  Args:
    embeddings_classifier: Keras model turning embeddings into logits.
    target_index: Choice of class index.
    **kwargs: Arguments passed on to search_embeddings_parallel.

  Returns:
    TopKSearchResults and all logits.
  """

  def classify_batch(batch, query_embedding_batch):
    del query_embedding_batch
    emb = batch[tf_examples.EMBEDDING]
    emb_shape = tf.shape(emb)
    flat_emb = tf.cast(tf.reshape(emb, [-1, emb_shape[-1]]), tf.float32)
    logits = embeddings_classifier(flat_emb)
    logits = tf.reshape(
        logits, [emb_shape[0], emb_shape[1], tf.shape(logits)[-1]]
    )
    # Restrict to target class.
    logits = logits[..., target_index]
    # Take the maximum logit over channels.
    logits = tf.reduce_max(logits, axis=-1)
    batch['scores'] = logits
    return batch

  return search_embeddings_parallel(
      score_fn=classify_batch, query_embedding_batch=None, **kwargs
  )
