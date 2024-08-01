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

"""Brute force search and reranking utilities."""

import concurrent
import threading
from typing import Any, Callable, Sequence

from chirp.projects.hoplite import interface
from chirp.projects.hoplite import search_results
import numpy as np


def worker_initializer(state):
  name = threading.current_thread().name
  state[name + 'db'] = state['db'].thread_split()


def brute_search_worker_fn(emb_ids: Sequence[int], state: dict[str, Any]):
  name = threading.current_thread().name
  emb_ids, embeddings = state[name + 'db'].get_embeddings(emb_ids)
  scores = state['score_fn'](embeddings, state['query_embedding'])
  top_locs = np.argpartition(scores, state['search_list_size'], axis=-1)
  return emb_ids[top_locs], scores[top_locs]


def threaded_brute_search(
    db: interface.GraphSearchDBInterface,
    query_embedding: np.ndarray,
    search_list_size: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    batch_size: int = 1024,
    max_workers: int = 8,
) -> tuple[search_results.TopKSearchResults, np.ndarray]:
  """Performs a brute-force search for neighbors of the query embedding.

  Args:
    db: Graph DB instance.
    query_embedding: Query embedding vector.
    search_list_size: Number of results to return.
    score_fn: Scoring function to use for ranking results.
    batch_size: Number of embeddings to score in each thread.
    max_workers: Maximum number of threads to use for the search.

  Returns:
    A TopKSearchResults object containing the search results, and a list of
    all scores computed during the search.
  """
  state = {}
  state['search_list_size'] = search_list_size
  state['db'] = db
  state['query_embedding'] = query_embedding
  state['score_fn'] = score_fn

  results = search_results.TopKSearchResults(search_list_size)
  # Commit the DB, since we are about to create views in multiple threads.
  db.commit()
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers,
      initializer=worker_initializer,
      initargs=(state,),
  ) as executor:
    ids = db.get_embedding_ids()
    futures = []
    for q in range(0, ids.shape[0], batch_size):
      futures.append(
          executor.submit(
              brute_search_worker_fn, ids[q : q + batch_size], state
          )
      )
    all_scores = []
    for f in futures:
      idxes, scores = f.result()
      all_scores.append(scores)
      for idx, score in zip(idxes, scores):
        if not results.will_filter(idx, score):
          results.update(
              search_results.SearchResult(idx, score), force_insert=True
          )
  all_scores = np.concatenate(all_scores)
  return results, all_scores


def brute_search(
    db: interface.GraphSearchDBInterface,
    query_embedding: np.ndarray,
    search_list_size: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[search_results.TopKSearchResults, np.ndarray]:
  """Performs a brute-force search for neighbors of the query embedding.

  Args:
    db: Graph DB instance.
    query_embedding: Query embedding vector.
    search_list_size: Number of results to return.
    score_fn: Scoring function to use for ranking results.

  Returns:
    A TopKSearchResults object containing the search results, and a list of
    all scores computed during the search.
  """
  results = search_results.TopKSearchResults(search_list_size)
  all_scores = []
  for idx in db.get_embedding_ids():
    target_embedding = db.get_embedding(idx)
    score = score_fn(query_embedding, target_embedding)
    all_scores.append(score)
    # Check filtering and then force insert to avoid creating a SearchResult
    # object for discarded objects. This saves a small amount of time in the
    # inner loop.
    if not results.will_filter(idx, score):
      results.update(search_results.SearchResult(idx, score), force_insert=True)
  return results, np.array(all_scores)


def rerank(
    query_embedding: np.ndarray,
    results: search_results.TopKSearchResults,
    db: interface.GraphSearchDBInterface,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> search_results.TopKSearchResults:
  """Rescore the search results using a different score function."""
  new_results = search_results.TopKSearchResults(results.top_k)
  for r in results:
    new_results.update(
        search_results.SearchResult(
            r.embedding_id,
            score_fn(query_embedding, db.get_embedding(r.embedding_id)),
        )
    )
  return new_results
