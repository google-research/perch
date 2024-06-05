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

"""Vamana implementation."""

import dataclasses
from typing import Callable

from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import search_results
import numpy as np
import tqdm


@dataclasses.dataclass
class HopliteSearchIndex:
  """Graph search using Vamana indexing."""

  db: interface.GraphSearchDBInterface
  dist: Callable[[np.ndarray, np.ndarray], float]

  @classmethod
  def from_db(
      cls, db: interface.GraphSearchDBInterface, metric_name: str = 'mip'
  ):
    """Create a VamanaSearchIndex from a GraphSearchDBInterface impl."""
    if metric_name == 'mip':
      dist = np.dot
    elif metric_name == 'cosine':
      dist = np.cos
    elif metric_name == 'euclidean':
      dist = lambda x, y: -np.linalg.norm(x - y)
    else:
      raise ValueError(f'Unknown metric name: {metric_name}')
    return cls(db, dist)

  def initialize_index(self, out_degree: int, seed: int = 42):
    self.db.drop_all_edges()

    # Create a random graph of degree R.
    graph_utils.add_random_edges(self.db, out_degree=out_degree, seed=seed)

  def get_quantile_bounds(self, degree_bound: int, alpha: float) -> np.ndarray:
    quant_bounds = np.linspace(1e-6, 1.0 - 1e-6, degree_bound)
    quant_bounds = np.exp(alpha * quant_bounds) - 1.0
    quant_bounds /= quant_bounds[-1] + 1e-6
    # Invert to focus on the top scores.
    quant_bounds = 1.0 - quant_bounds
    return quant_bounds

  def brute_search_initialize(
      self, out_degree: int, alpha: float, num_compares: int, seed: int = 42
  ):
    """Initialize graph edges with random search quantiles."""
    np.random.seed(seed)
    self.db.drop_all_edges()

    # Initialize the edges using quantiles from shuffled brute-force search.
    quant_bounds = self.get_quantile_bounds(out_degree, alpha)
    num_embeddings = self.db.count_embeddings()
    embedding_ids = self.db.get_embedding_ids()
    shuffled_embedding_ids = list(embedding_ids)
    np.random.shuffle(shuffled_embedding_ids)
    q = 0

    for idx in tqdm.tqdm(embedding_ids):
      query = self.db.get_embedding(idx)
      candidates = shuffled_embedding_ids[q : q + num_compares]
      embeddings = self.db.get_embeddings(candidates)
      scores = self.dist(embeddings, query)
      candidates = np.array(candidates, dtype=np.int32)

      candidates = tuple(candidates[c] for c in np.argsort(scores))
      keep_args = np.int32(len(candidates) * quant_bounds)
      p_out = set(candidates[c] for c in keep_args)
      for p_star in p_out:
        self.db.insert_edge(idx, int(p_star))
      q += out_degree
      if q + out_degree >= num_embeddings:
        q = 0
        np.random.shuffle(shuffled_embedding_ids)

  def greedy_search(
      self,
      query_embedding: np.ndarray,
      start_node: int,
      search_list_size: int = 100,
  ):
    """Apply the Vamana greedy search."""
    q_dist = lambda x: self.dist(query_embedding, x)

    visited = set()
    results = search_results.TopKSearchResults(search_list_size)

    # Insert start node into the TopKResults.
    start_node_embedding = self.db.get_embedding(start_node)
    start_score = q_dist(start_node_embedding)
    result = search_results.SearchResult(start_node, start_score)
    results.update(result)

    while True:
      for r in results:
        if r.embedding_id not in visited:
          visit_idx = r.embedding_id
          break
      else:
        break

      # Add the selected node to 'visited'.
      visited.add(visit_idx)

      # Add neighbors to the results and unvisited list.
      nbrs = self.db.get_edges(visit_idx)
      nbr_embeddings = self.db.get_embeddings(nbrs)
      nbr_scores = tuple(q_dist(e) for e in nbr_embeddings)
      for nbr_idx, nbr_score in zip(nbrs, nbr_scores):
        if nbr_idx in visited:
          continue
        if results.will_filter(nbr_idx, nbr_score):
          continue
        results.update(
            search_results.SearchResult(nbr_idx, nbr_score), force_insert=True
        )
    return results, visited

  def index(
      self,
      alpha: int,
      top_k: int,
      degree_bound: int,
      random_init_degree: int = 3,
      initialize: bool = False,
  ):
    """Create a search index over the DB."""
    if initialize:
      print('initializing...')
      self.initialize_index(random_init_degree)

    # Create a random ordering of the vectors.
    embedding_ids = list(self.db.get_embedding_ids())
    np.random.shuffle(embedding_ids)

    # TODO(tomdenton): get data medoid for initializing the search.
    for idx in tqdm.tqdm(embedding_ids):
      query_embedding = self.db.get_embedding(idx)
      unused_results, visited = self.greedy_search(
          query_embedding=query_embedding, start_node=1, search_list_size=top_k
      )
      self.robust_prune_vertex(idx, visited, alpha, degree_bound)

      # Check for edge size violations in neighbors of idx.
      nbrs = self.db.get_edges(idx)
      for nbr_idx in nbrs:
        candidates = set(self.db.get_edges(nbr_idx))
        if len(candidates) > degree_bound:
          candidates.add(idx)
          self.robust_prune_vertex(nbr_idx, candidates, alpha, degree_bound)
        else:
          self.db.insert_edge(nbr_idx, idx)

  def robust_prune_vertex_vect(
      self,
      idx: int,
      candidates: set[int],
      quant_bounds: np.ndarray,
      dry_run: bool = False,
  ):
    """Use quantiles to obtain well-distributed neighbors quickly."""
    out_edges = set(self.db.get_edges(idx))

    candidates = tuple(candidates.union(out_edges) - set((idx,)))
    if len(candidates) > quant_bounds.shape[0]:
      # Pruned edge set.
      p_embedding = self.db.get_embedding(idx)
      candidate_embeddings = self.db.get_embeddings(candidates)
      candidate_embeddings = np.stack(candidate_embeddings, axis=0)
      candidate_distances = np.array(
          [self.dist(p_embedding, e) for e in candidate_embeddings]
      )

      arg_sorted = np.argsort(candidate_distances)
      candidates = tuple(candidates[c] for c in arg_sorted)
      keep_args = len(candidates) * quant_bounds
      p_out = set(candidates[int(c)] for c in keep_args)
    else:
      p_out = candidates

    if dry_run:
      return p_out

    # Update the edge set.
    self.db.delete_edges(idx)
    for p_star in p_out:
      self.db.insert_edge(idx, p_star)

    return p_out

  def robust_prune_vertex(
      self,
      idx: int,
      candidates: set[int],
      alpha: float,
      degree_bound: int,
      dry_run: bool = False,
  ):
    """Computes a pruned edges set for the target index.."""
    out_edges = set(self.db.get_edges(idx))
    candidates = tuple(candidates.union(out_edges) - set((idx,)))

    # Pruned edge set.
    p_out = set()
    p_embedding = self.db.get_embedding(idx)

    candidate_embeddings = self.db.get_embeddings(candidates)
    candidate_distances = [
        self.dist(p_embedding, e) for e in candidate_embeddings
    ]
    candidate_triples = list(
        zip(candidate_distances, candidates, candidate_embeddings)
    )

    while candidate_triples:
      # get argmax of scores p, p*.
      candidate_triples = sorted(candidate_triples)
      unused_p_star_dist, p_star, p_star_emb = candidate_triples[-1]
      p_out.add(p_star)
      if len(p_out) >= degree_bound:
        break

      kill_list = set()
      for y_dist, y, y_emb in candidate_triples:
        dy = self.dist(p_star_emb, y_emb)
        # Derp, need to adjust this for max inner product...
        # if alpha * dy <= y_dist:
        if dy - alpha >= y_dist:
          kill_list.add(y)

      # Prune the candidates list.
      candidate_triples = [
          c for c in candidate_triples if c[1] not in kill_list
      ]

    if dry_run:
      return p_out

    # Update the edge set.
    self.db.delete_edges(idx)
    for p_star in p_out:
      self.db.insert_edge(idx, p_star)

    return p_out
