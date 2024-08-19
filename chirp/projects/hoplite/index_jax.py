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

"""Tooling for building a Hoplite index using GPU."""

import dataclasses
import itertools

import jax
from jax import numpy as jnp
import numpy as np


@dataclasses.dataclass
class IndexingData:
  step_num: int
  targets: jnp.ndarray
  candidates: jnp.ndarray
  embeddings: jnp.ndarray
  edges: jnp.ndarray
  delegate_lists: jnp.ndarray
  delegate_scores: jnp.ndarray
  batch_size: int
  alpha: float
  max_violations: int
  num_steps: int


# Note that this creates an import side-effect.
jax.tree_util.register_dataclass(
    IndexingData,
    data_fields=[
        'step_num',
        'targets',
        'candidates',
        'embeddings',
        'edges',
        'delegate_lists',
        'delegate_scores',
        'alpha',
        'max_violations',
    ],
    meta_fields=['batch_size', 'num_steps'],
)


def delegate_indexing(
    embeddings: jnp.ndarray,
    edges: jnp.ndarray,
    delegate_lists: jnp.ndarray | None,
    delegate_scores: jnp.ndarray | None,
    sample_size: int,
    max_delegates: int,
    alpha: float,
    max_violations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Create an edge set using delegated pruning.

  This is the main indexing method in this library, which uses the JIT-compiled
  prune_delegate_jax function as the inner-loop function to prune and surface
  delegates.

  Args:
    embeddings: The embedding matrix.
    edges: The edge matrix.
    delegate_lists: The delegate lists for each node.
    delegate_scores: The scores for each delegate.
    sample_size: The number of additional random candidates to score at each
      step.
    max_delegates: The maximum number of delegates to keep for each node.
    alpha: The pruning parameter.
    max_violations: Maximum number of pruning condition violations to allow.

  Returns:
    Final IndexingData.
  """
  corpus_size = embeddings.shape[0]
  targets = np.arange(corpus_size)
  np.random.shuffle(targets)
  candidates = np.arange(corpus_size)
  np.random.shuffle(candidates)
  if delegate_lists is None or delegate_scores is None:
    delegate_lists, delegate_scores = make_delegate_lists(
        corpus_size, max_delegates
    )

  batch_size = sample_size + edges.shape[1] + max_delegates
  initial_data = IndexingData(
      step_num=0,
      targets=targets,
      candidates=candidates,
      embeddings=embeddings,
      edges=edges,
      delegate_lists=delegate_lists,
      delegate_scores=delegate_scores,
      batch_size=batch_size,
      alpha=alpha,
      max_violations=max_violations,
      num_steps=corpus_size,
  )
  updated_data = unrolled_prune_delegate(initial_data)
  return updated_data


@jax.jit
def unrolled_prune_delegate(idx_data: IndexingData):
  """Wrapper for running prune_delegate_jax in a loop."""
  cond_fn = lambda idx_data: idx_data.step_num < idx_data.num_steps
  return jax.lax.while_loop(cond_fn, prune_delegate_jax, idx_data)


def prune_delegate_jax(idx_data: IndexingData):
  """Select a new set of edges for the target node."""
  target_idx = idx_data.targets[idx_data.step_num].astype(int)
  target_emb = idx_data.embeddings[target_idx]

  rolled_candidates = jnp.roll(idx_data.candidates, idx_data.batch_size + 37)
  random_candidates = rolled_candidates[: idx_data.batch_size]
  candidates = assemble_batch(
      target_idx,
      idx_data.edges[target_idx],
      idx_data.delegate_lists[target_idx],
      random_candidates,
  )
  candidate_embs = idx_data.embeddings[candidates]

  p_out, candidates, scores_c_c = prune_jax(
      target_emb,
      candidates,
      candidate_embs,
      idx_data.alpha,
      idx_data.max_violations,
  )

  # Update target edges and delete the old delegate list.
  degree_bound = idx_data.edges.shape[1]
  edges = idx_data.edges.at[target_idx].set(p_out[:degree_bound])

  # Update delegate lists with high-scoring elements.
  new_delegates, new_scores = update_delegates(
      idx_data.delegate_lists, idx_data.delegate_scores, candidates, scores_c_c
  )
  delegate_lists = idx_data.delegate_lists.at[candidates].set(new_delegates)
  delegate_scores = idx_data.delegate_scores.at[candidates].set(new_scores)

  return IndexingData(
      step_num=idx_data.step_num + 1,
      targets=idx_data.targets,
      candidates=rolled_candidates,
      embeddings=idx_data.embeddings,
      edges=edges,
      delegate_lists=delegate_lists,
      delegate_scores=delegate_scores,
      batch_size=idx_data.batch_size,
      alpha=idx_data.alpha,
      max_violations=idx_data.max_violations,
      num_steps=idx_data.num_steps,
  )


def assemble_batch(
    target_idx: jnp.ndarray,
    target_edges: jnp.ndarray,
    target_delegates: jnp.ndarray,
    random_candidates: jnp.ndarray,
):
  """Assemble a batch of candidates for the target node."""
  max_edges = target_edges.shape[0]
  max_dels = target_delegates.shape[0]
  joined = jnp.concatenate([target_edges, target_delegates], axis=0)
  joined = unique1d(joined)
  joined_mask = jnp.logical_and(joined >= 0, joined != target_idx)
  joined_candidates = random_candidates[: max_edges + max_dels]
  joined_candidates = jnp.where(joined_mask, joined, joined_candidates)

  remainder = random_candidates[max_edges + max_dels :]
  return jnp.concatenate([joined_candidates, remainder], axis=0)


def update_delegates(
    delegate_lists: jnp.ndarray,
    delegate_scores: jnp.ndarray,
    candidates: jnp.ndarray,
    candidate_scores: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Update the delegate lists with the highest-scoring candidates."""
  max_delegates = delegate_lists.shape[1]

  prev_scores = delegate_scores[candidates, :]
  safe_scores = jnp.fill_diagonal(candidate_scores, -jnp.inf, inplace=False)
  combined_scores = jnp.concatenate([prev_scores, safe_scores], axis=1)
  stacked_candidates = jnp.tile(
      candidates[jnp.newaxis, :], [candidates.shape[0], 1]
  )
  combined_delegates = jnp.concatenate(
      [delegate_lists[candidates], stacked_candidates], axis=1
  )

  # Eliminate repeated delegates.
  combined_delegates = unique1d(combined_delegates)
  combined_scores = jnp.where(
      combined_delegates == -1, -jnp.inf, combined_scores
  )

  # Sort the combined delegates by score.
  combined_scores, combined_delegates = cosort(  # pylint: disable=unbalanced-tuple-unpacking
      combined_scores, combined_delegates, descending=True
  )
  new_scores = combined_scores[:, :max_delegates]
  new_delegates = combined_delegates[:, :max_delegates]
  return new_delegates, new_scores


def prune_jax(
    target_emb: jnp.ndarray,
    candidates: jnp.ndarray,
    candidate_embs: jnp.ndarray,
    alpha: float,
    max_violations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Prune a set of candidates for the target embedding."""
  scores_c_t = jnp.dot(candidate_embs, target_emb)

  scores_c_t, candidates, candidate_embs = cosort(
      scores_c_t, candidates, candidate_embs, axis=0, descending=True
  )
  scores_c_c = jnp.tensordot(candidate_embs, candidate_embs, axes=(-1, -1))

  # Sparse neighborhood condition.
  mask = scores_c_c >= scores_c_t[np.newaxis, :] + alpha
  mask = jnp.triu(mask, k=1)
  violation_counts = mask.sum(axis=0)
  violation_mask = violation_counts > max_violations

  _, p_out, violation_mask = cosort(  # pylint: disable=unbalanced-tuple-unpacking
      violation_counts, candidates, violation_mask, axis=-1
  )

  empty_edges = -1 * jnp.ones_like(p_out)
  p_out = jnp.where(violation_mask, empty_edges, p_out)
  return p_out, candidates, scores_c_c


def make_delegate_lists(
    num_embeddings: int, max_delegates: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
  delegate_lists = -1 * jnp.ones([num_embeddings, max_delegates], jnp.int32)
  delegate_scores = -jnp.inf * jnp.ones(
      [num_embeddings, max_delegates], jnp.float16
  )
  return delegate_lists, delegate_scores


def unique1d(v: jnp.ndarray) -> jnp.ndarray:
  """Deduplicate v along the last axis, replacing dupes with -1.

  For example, if v = [1, 2, 3, 1, 4, 2, 5],
  then unique1d(v) = [1, 2, 3, -1, 4, -1, 5].
  Obviously, it's a bad idea to use this if -1 might appear in the array.

  Args:
    v: an array of values.

  Returns:
    v with duplicates replaced by -1.
  """
  v_sort_locs = jnp.argsort(v, axis=-1)
  v_sorted = jnp.take_along_axis(v, v_sort_locs, axis=-1)
  dv_sorted = jnp.concatenate(
      [
          jnp.ones_like(v[..., :1]),  # keep first even if diff === 0.
          jnp.diff(v_sorted),
      ],
      axis=-1,
  )
  masked = jnp.where(dv_sorted == 0, -1, v_sorted)
  inverse_perm = jnp.argsort(v_sort_locs, axis=-1)
  return jnp.take_along_axis(masked, inverse_perm, axis=-1)


def cosort(
    s: jnp.ndarray, *others, axis: int = -1, descending: bool = False
) -> tuple[jnp.ndarray, ...]:
  """Sort s and others by the values of s."""
  sort_locs = jnp.argsort(s, axis=axis, descending=descending)
  if len(s.shape) == 1 and axis == 0:
    return tuple((a[sort_locs] for a in itertools.chain((s,), others)))

  return tuple((
      jnp.take_along_axis(a, sort_locs, axis=axis)
      for a in itertools.chain((s,), others)
  ))
