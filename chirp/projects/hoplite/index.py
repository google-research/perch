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

import collections
import dataclasses
from typing import Callable

from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import score_functions
from chirp.projects.hoplite import search_results
import numpy as np
import tqdm


@dataclasses.dataclass
class HopliteSearchIndex:
  """Graph search using Vamana indexing."""

  db: interface.GraphSearchDBInterface
  score_fn: Callable[[np.ndarray, np.ndarray], float | np.ndarray]

  @classmethod
  def from_db(
      cls, db: interface.GraphSearchDBInterface, score_fn_name: str = 'dot'
  ) -> 'HopliteSearchIndex':
    """Create a VamanaSearchIndex from a GraphSearchDBInterface impl."""
    # TODO(tomdenton): Use an enum for metric_name.
    if score_fn_name in ('mip', 'dot'):
      # mip == Max Inner Prouct
      score_fn = score_functions.numpy_dot
    elif score_fn_name in ('jax_mip', 'jax_dot'):
      score_fn = score_functions.get_jax_dot()
    elif score_fn_name == 'cosine':
      score_fn = score_functions.numpy_cos
    elif score_fn_name == 'euclidean':
      score_fn = score_functions.numpy_euclidean
    else:
      raise ValueError(f'Unknown metric name: {score_fn_name}')
    return cls(db, score_fn=score_fn)

  def initialize_index(self, out_degree: int, seed: int = 42) -> None:
    self.db.drop_all_edges()

    # Initialize with a random graph of degree out_degree.
    graph_utils.add_random_edges(self.db, out_degree=out_degree, seed=seed)

  def brute_prune_initialize(
      self,
      target_degree: int,
      alpha: float,
      num_compares: int,
      seed: int = 42,
      add_reverse_edges: bool = True,
      pad_edges: bool = True,
  ):
    """Initialize graph edges by pruning a random set of vectors for each node.

    Args:
      target_degree: Maximum number of edges per node for initialized graph.
      alpha: Vamana graph indexing hyperparameter.
      num_compares: Number of random sample nodes to collect for each node.
      seed: Random seed.
      add_reverse_edges: Whether to add reverse edges opportunistically.
      pad_edges: Whether to pad the edge set with random edges to reach
        target_degree. This is applied after reverse_edges.
    """
    scan_rng, pad_rng = np.random.default_rng(seed=seed).spawn(2)
    self.db.drop_all_edges()

    # Initialize the edges using quantiles from shuffled brute-force search.
    embedding_ids = self.db.get_embedding_ids()
    random_id_generator = graph_utils.random_batched_iterator(
        embedding_ids, batch_size=num_compares, rng=scan_rng
    )
    for idx in tqdm.tqdm(embedding_ids):
      candidates = next(random_id_generator)
      p_out = self.robust_prune_vertex(idx, candidates, alpha, target_degree)
      self.db.insert_edges(idx, p_out)

      if add_reverse_edges:
        for nbr_idx in p_out:
          if len(self.db.get_edges(nbr_idx)) < target_degree:
            self.db.insert_edge(nbr_idx, idx)

    if add_reverse_edges:
      # Remove duplicate edges, if any.
      for idx in embedding_ids:
        nbrs = self.db.get_edges(idx)
        deduped_nbrs = np.unique(nbrs)
        if len(nbrs) > len(deduped_nbrs):
          self.db.delete_edges(idx)
          self.db.insert_edges(idx, deduped_nbrs)

    if pad_edges:
      # Add a random set of edges to each vertex to reach target degree.
      random_padding_generator = graph_utils.random_batched_iterator(
          embedding_ids, batch_size=target_degree, rng=pad_rng
      )

      for idx in tqdm.tqdm(embedding_ids):
        edges = self.db.get_edges(idx)
        pad_amount = target_degree - edges.shape[0]
        if pad_amount <= 0:
          continue
        candidates = next(random_padding_generator)
        candidates = np.setdiff1d(candidates, edges)[:pad_amount]
        self.db.insert_edges(idx, candidates)

  def greedy_search(
      self,
      query_embedding: np.ndarray,
      start_node: int,
      search_list_size: int = 100,
      deterministic: bool = False,
      max_visits: int | None = None,
  ) -> tuple[search_results.TopKSearchResults, np.ndarray]:
    """Apply the Vamana greedy search.

    Args:
      query_embedding: Embedding to search for.
      start_node: Entry node index.
      search_list_size: Top-k value for search.
      deterministic: Ensure that the search path is fully reproducible.
      max_visits: Visit no more than this many nodes.

    Returns:
      The TopKSearchResults and the sequence of all 'visited' nodes.
    """
    visited = {}
    results = search_results.TopKSearchResults(search_list_size)

    # Insert start node into the TopKResults.
    start_node_embedding = self.db.get_embedding(start_node)
    start_score = self.score_fn(start_node_embedding, query_embedding)
    result = search_results.SearchResult(start_node, start_score)
    results.update(result)

    while max_visits is None or len(visited) < max_visits:
      # Get the best result we have not yet visited.
      for r in results:
        if r.embedding_id not in visited:
          visit_idx = r.embedding_id
          break
      else:
        break

      # Add the selected node to 'visited'.
      visited[visit_idx] = None

      # We will examine neighbors of the visited node.
      nbrs = self.db.get_edges(visit_idx)
      # Filter visited neighbors.
      nbrs = nbrs[np.array(tuple(n not in visited for n in nbrs), dtype=bool)]

      nbrs, nbr_embeddings = self.db.get_embeddings(nbrs)
      if deterministic:
        order = np.argsort(nbrs)
        nbr_embeddings = nbr_embeddings[order]
        nbrs = nbrs[order]
      nbr_scores = self.score_fn(nbr_embeddings, query_embedding)

      if len(results.search_results) >= search_list_size:
        # Drop any elements bigger than the current result set's min_score.
        keep_args = np.where(nbr_scores >= results.min_score)
        nbrs = nbrs[keep_args]
        nbr_scores = nbr_scores[keep_args]

      for nbr_idx, nbr_score in zip(nbrs, nbr_scores):
        if results.will_filter(nbr_idx, nbr_score):
          continue
        results.update(
            search_results.SearchResult(nbr_idx, nbr_score), force_insert=True
        )
    return results, np.array(tuple(visited.keys()))

  def index(
      self,
      alpha: int,
      top_k: int,
      degree_bound: int,
      start_node: int = -1,
      initialize: bool = False,
      random_init_degree: int = 3,
  ) -> None:
    """Create a search index over the DB.

    Implements the Vamana algorithm, producing a graph over the vectors in the
    database which optimizes greedy_search.

    Args:
      alpha: Hyperparameter controlling approximate degree of the graph.
      top_k: Search depth used when indexing.
      degree_bound: Maximum allowe degree for each node in the graph.
      start_node: Index of embedding used as the root for all searches.
      initialize: Whether to drop all edges and initialize with random edges.
      random_init_degree: If initializing the graph, initializes to this random
        degree.
    """
    if initialize:
      print('initializing...')
      self.initialize_index(random_init_degree)

    if start_node < 0:
      start_node = self.db.get_one_embedding_id()

    # Create a random ordering of the vectors.
    embedding_ids = list(self.db.get_embedding_ids())
    np.random.shuffle(embedding_ids)

    # TODO(tomdenton): get data medoid for initializing the search.
    for idx in tqdm.tqdm(embedding_ids):
      query_embedding = self.db.get_embedding(idx)
      _, visited = self.greedy_search(
          query_embedding=query_embedding,
          start_node=start_node,
          search_list_size=top_k,
      )
      p_out = self.robust_prune_vertex(idx, visited, alpha, degree_bound)
      self.db.delete_edges(idx)
      self.db.insert_edges(idx, p_out)

      # Check for edge size violations in neighbors of idx.
      nbrs = self.db.get_edges(idx)
      for nbr_idx in nbrs:
        candidates = self.db.get_edges(nbr_idx)
        if len(candidates) >= degree_bound:
          if idx not in candidates:
            candidates = np.append(candidates, idx)
          p_out = self.robust_prune_vertex(
              nbr_idx, candidates, alpha, degree_bound
          )
          self.db.delete_edges(nbr_idx)
          self.db.insert_edges(nbr_idx, p_out)
        else:
          self.db.insert_edge(nbr_idx, idx)
    self.db.commit()

  def index_delegates(
      self,
      degree_bound: int,
      alpha: float = 1.0,
      num_tree_iterations: int = 3,
  ):
    """Create an index using delegated pruning trees."""
    root_node = self.db.get_one_embedding_id()
    self.db.drop_all_edges()

    # First create a single tree using delegate pruning.
    self.index_delegates_single(
        root_node=root_node,
        degree_bound=degree_bound,
        alpha=alpha,
    )
    print(f'Root node degree is {len(self.db.get_edges(root_node))}.')
    print('Adding reverse edges...')
    self.add_reverse_edges(degree_bound)
    self.dedupe_edges()

    visited = [root_node]

    for _ in tqdm.tqdm(range(num_tree_iterations - 1)):
      # Choose the next 'root' by choosing a node with low degree which we
      # have not used as a root yet.
      node_degrees = sorted([
          (len(np.unique(self.db.get_edges(idx))), idx)
          for idx in self.db.get_embedding_ids()
      ])
      for _, new_root in node_degrees:
        if new_root not in visited:
          break
      else:
        # So long as the number of iterations is not too high, this should
        # never happen.
        raise AssertionError('No new root found.')

      visited.append(new_root)
      self.index_delegates_single(
          root_node=new_root,
          degree_bound=degree_bound,
          alpha=alpha,
      )
      if len(visited) > num_tree_iterations:
        break
    edge_count = self.db.count_edges()
    print(f'\nGraph has {edge_count} internal edges.')
    print('Adding reverse edges...')
    self.add_reverse_edges(degree_bound)
    self.dedupe_edges()
    return visited

  def add_reverse_edges(self, degree_bound: int):
    for r in self.db.get_embedding_ids():
      for nbr in np.unique(self.db.get_edges(r)):
        nbr_edges = self.db.get_edges(nbr)
        if nbr_edges.shape[0] < degree_bound and r not in nbr_edges:
          self.db.insert_edge(nbr, r)

  def dedupe_edges(self):
    for r in self.db.get_embedding_ids():
      updated_edges = np.unique(self.db.get_edges(r))
      self.db.delete_edges(r)
      self.db.insert_edges(r, updated_edges)

  def index_delegates_single(
      self,
      root_node: int,
      degree_bound: int,
      alpha: float = 1.0,
  ):
    """Insert edges of a random (degree_bound)-ary tree with given root node."""
    candidates = self.db.get_embedding_ids()
    delegate_sets = {root_node: candidates}
    visited = set()

    while delegate_sets:
      target, candidates = delegate_sets.popitem()
      if target in visited:
        continue
      visited.add(target)
      if not candidates.shape[0]:
        continue

      target_edges = self.db.get_edges(target)
      if candidates.shape[0] + target_edges.shape[0] <= degree_bound:
        # Instead of pruning low-degree nodes, just add the edges.
        new_edges = np.setdiff1d(candidates, target_edges)
        self.db.insert_edges(target, new_edges)
        continue

      p_out = self.robust_prune_vertex(
          target,
          candidates,
          alpha=alpha,
          degree_bound=degree_bound,
      )
      new_delegate_sets = self.assign_delegates(p_out, candidates)

      self.db.delete_edges(target)
      self.db.insert_edges(target, p_out)
      delegate_sets.update(new_delegate_sets)

  def assign_delegates(self, targets: np.ndarray, candidates: np.ndarray):
    """Assign each candidate to the target with the highest score.

    Args:
      targets: The nodes to delegate to.
      candidates: The nodes to assign to the targets.

    Returns:
      A dict mapping each target to the set of candidates assigned to it.
    """
    if not targets.shape[0]:
      return dict()
    candidates = np.setdiff1d(candidates, targets)
    targets, target_embeddings = self.db.get_embeddings(targets)
    candidates, candidate_embeddings = self.db.get_embeddings(candidates)
    scores = self.score_fn(candidate_embeddings, target_embeddings)
    delegations = np.argmax(scores, axis=1)
    delegations = targets[delegations]
    delegate_sets = collections.defaultdict(list)
    for delegated, target in zip(candidates, delegations):
      delegate_sets[target].append(delegated)
    return {k: np.array(v) for k, v in delegate_sets.items()}

  def robust_prune_vertex(
      self,
      idx: int,
      candidates: np.ndarray,
      alpha: float,
      degree_bound: int,
  ) -> np.ndarray:
    """Computes a pruned edges set for the target index."""
    out_edges = self.db.get_edges(idx)
    candidates = np.concatenate([candidates, out_edges], axis=0)
    candidates = np.unique(candidates).astype(np.int32)
    # Delete idx if it appears in the candidates.
    candidates = candidates[candidates != idx]

    # Pruned edge set.
    p_out = set()
    p_embedding = self.db.get_embedding(idx)

    candidates, candidate_embeddings = self.db.get_embeddings(candidates)
    candidate_scores = self.score_fn(candidate_embeddings, p_embedding)

    # Sort the candidates by score so that the last element is always the best.
    sort_locs = np.argsort(candidate_scores)
    candidate_scores = candidate_scores[sort_locs]
    candidate_embeddings = candidate_embeddings[sort_locs]
    candidates = candidates[sort_locs]

    while candidates.shape[0] and len(p_out) < degree_bound:
      # get argmax of scores p, p*.
      p_star_j = -1
      p_out.add(candidates[p_star_j])
      if len(p_out) >= degree_bound:
        break

      p_star_emb = candidate_embeddings[p_star_j]
      p_star_y_score = np.array(self.score_fn(candidate_embeddings, p_star_emb))
      # TODO(tomdenton): Allow configuration of the masking criterium.
      # Original criteria, using euclidean distance, where down is good:
      # alpha * d(p_star, y) <= d(p, y)
      # 'if y is a better result for p_star than for p, we can discard y.'
      # But up is good for us, and dot products may be negative.
      # If we consider the dot product similarity to be on a logarithmic scale,
      # then iteratively subtracting ln(alpha) is equivalent to iterative
      # division, as in the original euclidean distance case.
      mask = p_star_y_score < candidate_scores + alpha
      # Always drop p_star_j, since it is added to p_out.
      mask[p_star_j] = False

      # Prune the candidates list.
      candidates = candidates[mask]
      candidate_scores = candidate_scores[mask]
      candidate_embeddings = candidate_embeddings[mask]
    return np.array(tuple(p_out))

  def test_recall(
      self,
      query: np.ndarray,
      search_fn: Callable[[np.ndarray], search_results.TopKSearchResults],
      eval_top_k: int,
      verbose=False,
  ) -> tuple[float, set[int], set[int]]:
    """Check recall@eval_top_k for greedy search vs brute-force search.

    Args:
      query: The query embedding to search for.
      search_fn: A function that takes a query embedding and returns a
        TopKSearchResults object.
      eval_top_k: The number of results to evaluate for recall.
      verbose: Whether to print the recall value.

    Returns:
      The recall value, the set of id's found by the search function and
      ground-truth ids from the brute-force search.
    """
    graph_results = search_fn(query)
    graph_keys = set(r.embedding_id for r in graph_results)

    brute_results, _ = graph_utils.brute_search(
        self.db, query, search_list_size=eval_top_k, score_fn=self.score_fn
    )
    brute_keys = set(r.embedding_id for r in brute_results)
    recall = len(brute_keys.intersection(graph_keys)) / len(brute_keys)
    if verbose:
      print(f'recall@{eval_top_k}      : {recall}', flush=True)
    return recall, graph_keys, brute_keys

  def multi_test_recall(
      self,
      search_fn,
      eval_top_k: int = 32,
      num_runs: int = 100,
      disable_tqdm: bool = True,
  ) -> float:
    """Test average recall compared to brute search over multiple queries."""
    recalls = []
    emb_dim = self.db.embedding_dimension()
    rng = np.random.default_rng(seed=22)
    node_iterator = graph_utils.random_batched_iterator(
        self.db.get_embedding_ids(), batch_size=1, rng=rng
    )
    for _ in tqdm.tqdm(range(num_runs), disable=disable_tqdm):
      # Create a realistic random query by adding noise to an existing vector.
      # TODO(tomdenton): Consider measuring an appropriate noise scale.
      noise = np.random.normal(size=emb_dim, loc=0.0, scale=1e-5)
      [embedding_id] = next(node_iterator)
      query = self.db.get_embedding(embedding_id) + noise
      recall, _, _ = self.test_recall(
          query,
          search_fn=search_fn,
          eval_top_k=eval_top_k,
          verbose=False,
      )
      recalls.append(recall)
    return float(np.mean(recalls))
