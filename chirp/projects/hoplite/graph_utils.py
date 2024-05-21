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

"""Utility functions for testing and benchmarking."""

from collections.abc import Callable

from chirp.projects.hoplite import interface
from chirp.projects.hoplite import search_results
import numpy as np
import tqdm


def insert_random_embeddings(
    db: interface.GraphSearchDBInterface,
    emb_dim: int = 1280,
    num_embeddings: int = 1000,
    seed: int = 42,
):
  """Insert randomly generated embedding vectors into the DB."""
  rng = np.random.default_rng(seed=seed)
  for i in tqdm.tqdm(range(num_embeddings)):
    embedding = np.float32(rng.normal(size=emb_dim, loc=0, scale=1.0))
    metadata = {'file_id': 'random_file', 'offset_s': 1.0 * i}
    db.insert_embedding(embedding, **metadata)
  db.commit()


def connected_components(graph_db: interface.GraphSearchDBInterface):
  """Compute the connected components of the graph."""
  components = []
  component = set()
  visited = set()
  pool = set([1])
  idxes = set(range(1, graph_db.count_embeddings() + 1))
  for _ in tqdm.tqdm(range(graph_db.count_embeddings() + 1)):
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
        components.append(component)
        component = set()
        pool = set([tuple(unvisited)[0]])
  return components


def add_random_edges(
    graph_db: interface.GraphSearchDBInterface, out_degree: int, seed: int = 42
):
  """Add a random connected subgraph to the DB.

  Creates a single cycle, ensuring connectedness, and adds random edges to meet
  target out_degree.

  Args:
    graph_db: Graph DB instance.
    out_degree: Target number of outgoing edges per node.
    seed: Random seed.
  """
  num_embeddings = graph_db.count_embeddings()
  np.random.seed(seed)
  embedding_ids = list(range(1, num_embeddings + 1))
  q = 0
  # Random choice is a bottleneck...
  # Instead, shuffle and use permuted indices, then re-shuffle when we get
  # near the end of the list.
  # This gets throughput of ~18k edges/sec, compared to ~300 edges/sec when
  # using np.random.choice(embedding_ids, 10)... 60x speedup.
  np.random.shuffle(embedding_ids)
  cyclic_order = embedding_ids[:]
  # We get a 'free' edge by creating an initial cycle (ensuring graph
  # connectivity). Reduce out_degree by one to reflect this.
  out_degree -= 1
  for i in tqdm.tqdm(range(1, num_embeddings + 1)):
    cyclic_edge = cyclic_order[(i + 1) % num_embeddings]
    outs = embedding_ids[q : q + out_degree]
    if cyclic_edge in outs:
      outs = embedding_ids[q : q + out_degree + 1]
      q += out_degree + 1
    else:
      outs.append(cyclic_edge)
      q += out_degree

    if q + out_degree >= num_embeddings:
      q = 0
      np.random.shuffle(embedding_ids)
    for j in outs:
      if j == i:
        graph_db.insert_edge(i, num_embeddings)
      else:
        graph_db.insert_edge(i, j)
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
):
  """Performs a brute-force search for neighbors of the query embedding."""
  results = search_results.TopKSearchResults(search_list_size)
  for idx in db.get_embedding_ids():
    target_embedding = db.get_embedding(idx)
    score = score_fn(query_embedding, target_embedding)
    if not results.will_filter(idx, score):
      results.update(search_results.SearchResult(idx, score))
  return results
