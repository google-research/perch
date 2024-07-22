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

"""Utility functions for graph operations."""

from collections.abc import Callable, Iterator

from chirp.projects.hoplite import interface
from chirp.projects.hoplite import search_results
import numpy as np
import tqdm


def random_batched_iterator(
    ids: np.ndarray,
    batch_size: int,
    rng: np.random.RandomState,
) -> Iterator[np.ndarray]:
  """Yields batches embedding ids, shuffled after each of unlimited epochs."""
  rng.shuffle(ids)
  q = 0
  while True:
    if q + batch_size >= len(ids):
      q = 0
      rng.shuffle(ids)
    yield ids[q : q + batch_size]
    q += batch_size


def insert_random_embeddings(
    db: interface.GraphSearchDBInterface,
    emb_dim: int = 1280,
    num_embeddings: int = 1000,
    seed: int = 42,
):
  """Insert randomly generated embedding vectors into the DB."""
  rng = np.random.default_rng(seed=seed)
  np_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
  dataset_names = ('a', 'b', 'c')
  for _ in tqdm.tqdm(range(num_embeddings)):
    embedding = np.float32(rng.normal(size=emb_dim, loc=0, scale=1.0))
    dataset_name = rng.choice(dataset_names)
    source_name = ''.join(
        [str(a) for a in np.random.choice(np_alpha, size=3, replace=False)]
    )
    offsets = rng.integers(0, 100, size=[1])
    source = interface.EmbeddingSource(dataset_name, source_name, offsets)
    db.insert_embedding(embedding, source)
  db.commit()


def connected_components(
    graph_db: interface.GraphSearchDBInterface,
) -> list[set[int]]:
  """Compute the connected components of the graph."""
  visited = set()
  pool = set([graph_db.get_one_embedding_id()])

  component = pool.copy()
  components = [component]
  idxes = set(graph_db.get_embedding_ids())
  for _ in tqdm.tqdm(range(1, graph_db.count_embeddings())):
    idx = pool.pop()
    visited.add(idx)
    for e in graph_db.get_edges(idx):
      if e not in visited:
        pool.add(e)
        component.add(e)
    if not pool:
      unvisited = idxes.difference(visited)
      if not unvisited:
        break
      else:
        pool = set([next(iter(unvisited))])
        component = pool.copy()
        components.append(component)
  return components


def add_random_edges(
    graph_db: interface.GraphSearchDBInterface, out_degree: int, seed: int = 42
):
  """Add edges to form a random connected graph to the DB.

  Creates a single cycle, ensuring connectedness, and adds random edges to meet
  target out_degree.

  Args:
    graph_db: Graph DB instance.
    out_degree: Target number of outgoing edges per node.
    seed: Random seed.
  """
  num_embeddings = graph_db.count_embeddings()
  np.random.seed(seed)
  embedding_ids = graph_db.get_embedding_ids()

  # Random choice is a bottleneck...
  # Instead, shuffle and use permuted indices, then re-shuffle when we get
  # near the end of the list.
  # This gets throughput of ~18k edges/sec, compared to ~300 edges/sec when
  # using np.random.choice(embedding_ids, 10)... 60x speedup.
  np.random.shuffle(embedding_ids)
  cyclic_order = embedding_ids.copy()
  # We get a 'free' edge by creating an initial cycle (ensuring graph
  # connectivity). Reduce out_degree by one to reflect this.
  out_degree -= 1
  # q is an index into the shuffled id's for fast selection of a random subset.
  q = 0
  for idx in tqdm.tqdm(embedding_ids):
    cyclic_edge = cyclic_order[idx % num_embeddings]
    outs = embedding_ids[q : q + out_degree]
    if cyclic_edge in outs:
      outs = embedding_ids[q : q + out_degree + 1]
      q += out_degree + 1
    else:
      outs = np.concatenate([outs, [cyclic_edge]])
      q += out_degree
    graph_db.insert_edges(idx, outs)

    if q + out_degree + 1 >= num_embeddings:
      q = 0
      np.random.shuffle(embedding_ids)
  graph_db.commit()


def random_walk(
    db: interface.GraphSearchDBInterface,
    start_idx: int = 1,
    steps: int = 100,
    seed: int = 42,
    fetch_embeddings: bool = False,
):
  """Perform a random walk from start_idx."""
  # This is just for benchmarking...
  idx = start_idx
  rng = np.random.default_rng(seed=seed)
  for _ in tqdm.tqdm(range(steps)):
    # get outgoing edges for the current index.
    edges = db.get_edges(idx)
    if fetch_embeddings:
      db.get_embeddings(edges)
    if not edges:
      print('No edges found for index %d', idx)
      break
    idx = edges[rng.integers(0, len(edges))]
  return idx


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
