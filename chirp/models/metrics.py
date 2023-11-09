# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Metrics for training and validation."""

from typing import Any, Dict

from jax import lax
from jax import numpy as jnp
from jax import scipy


def map_(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    sort_descending: bool = True,
) -> jnp.ndarray:
  return average_precision(
      scores=logits,
      labels=labels,
      label_mask=label_mask,
      sort_descending=sort_descending,
  )


def cmap(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    sort_descending: bool = True,
    sample_threshold: int = 0,
) -> Dict[str, Any]:
  """Class mean average precision."""
  class_aps = average_precision(
      scores=logits.T, labels=labels.T, sort_descending=sort_descending
  )
  mask = jnp.sum(labels, axis=0) > sample_threshold
  class_aps = jnp.where(mask, class_aps, jnp.nan)
  macro_cmap = jnp.mean(class_aps, where=mask)
  return {
      'macro': macro_cmap,
      'individual': class_aps,
  }


def roc_auc(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    sort_descending: bool = True,
    sample_threshold: int = 1,
) -> Dict[str, Any]:
  """Computes ROC-AUC scores.

  Args:
    logits: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes average_precision on descendingly sorted
      inputs.
    sample_threshold: Only classes with at least this many samples will be used
      in the calculation of the final metric. By default this is 1, which means
      that classes without any positive examples will be ignored.

  Returns:
    A dictionary of ROC-AUC scores using the arithmetic ('macro') and
    geometric means, along with individual class ('individual') ROC-AUC and its
    variance.
  """
  if label_mask is not None:
    label_mask = label_mask.T
  class_roc_auc, class_roc_auc_var = generalized_mean_rank(
      logits.T, labels.T, label_mask=label_mask, sort_descending=sort_descending
  )
  mask = jnp.sum(labels, axis=0) >= sample_threshold
  class_roc_auc = jnp.where(mask, class_roc_auc, jnp.nan)
  class_roc_auc_var = jnp.where(mask, class_roc_auc_var, jnp.nan)
  return {
      'macro': jnp.mean(class_roc_auc, where=mask),
      'geometric': jnp.exp(jnp.mean(jnp.log(class_roc_auc), where=mask)),
      'individual': class_roc_auc,
      'individual_var': class_roc_auc_var,
  }


def negative_snr_loss(
    source: jnp.ndarray,
    estimate: jnp.ndarray,
    max_snr: float = 1e6,
    eps: float = 1e-8,
) -> jnp.ndarray:
  """Negative SNR loss with max SNR term.

  Args:
    source: Groundtruth signal.
    estimate: Estimated signal.
    max_snr: SNR threshold which minimizes loss. The default 1e6 yields an
      unbiased SNR calculation.
    eps: Log stabilization epsilon.

  Returns:
    Loss tensor.
  """
  snrfactor = 10.0 ** (-max_snr / 10.0)
  ref_pow = jnp.sum(jnp.square(source), axis=-1)
  bias = snrfactor * ref_pow

  numer = 10.0 * jnp.log10(ref_pow + eps)
  err_pow = jnp.sum(jnp.square(source - estimate), axis=-1)
  return 10 * jnp.log10(bias + err_pow + eps) - numer


def average_precision(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    sort_descending: bool = True,
    interpolated: bool = False,
) -> jnp.ndarray:
  """Average precision.

  The average precision is the area under the precision-recall curve. When
  using interpolation we take the maximum precision over all smaller recalls.
  The intuition is that it often makes sense to evaluate more documents if the
  total percentage of relevant documents increases.
  Average precision is computed over the last axis.

  Args:
    scores: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes average_precision on descendingly sorted
      inputs.
    interpolated: Whether to use interpolation.

  Returns:
    The average precision.
  """
  if label_mask is not None:
    # Set all masked labels to zero, and send the scores for those labels to a
    # low/high value (depending on whether we sort in descending order or not).
    # Then the masked scores+labels will not impact the average precision
    # calculation.
    labels = labels * label_mask
    extremum_score = (
        jnp.min(scores) - 1.0 if sort_descending else jnp.max(scores) + 1.0
    )
    scores = jnp.where(label_mask, scores, extremum_score)
  idx = jnp.argsort(scores)
  if sort_descending:
    idx = jnp.flip(idx, axis=-1)
  scores = jnp.take_along_axis(scores, idx, axis=-1)
  labels = jnp.take_along_axis(labels, idx, axis=-1)
  pr_curve = jnp.cumsum(labels, axis=-1) / (jnp.arange(labels.shape[-1]) + 1)
  if interpolated:
    pr_curve = lax.cummax(pr_curve, reverse=True, axis=-1)

  # In case of an empty row, assign precision = 0, and avoid dividing by zero.
  mask = jnp.float32(jnp.sum(labels, axis=-1) != 0)
  raw_av_prec = jnp.sum(pr_curve * labels, axis=-1) / jnp.maximum(
      jnp.sum(labels, axis=-1), 1.0
  )
  return mask * raw_av_prec


def generalized_mean_rank(
    scores: jnp.ndarray,
    labels: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    sort_descending: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the generalized mean rank and its variance over the last axis.

  The generalized mean rank can be expressed as

      (sum_i #P ranked above N_i) / (#P * #N),

  or equivalently,

      1 - (sum_i #N ranked above P_i) / (#P * #N).

  This metric is usually better visualized in the logits domain, where it
  reflects the log-odds of ranking a randomly-chosen positive higher than a
  randomly-chosen negative.

  Args:
    scores: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes the generalize mean rank on descendingly
      sorted inputs.

  Returns:
    The generalized mean rank and its variance. The variance is calculated by
    considering each positive to be an independent sample of the value
    1 - #N ranked above P_i / #N. This gives a measure of how consistently
    positives are ranked.
  """
  idx = jnp.argsort(scores, axis=-1)
  if sort_descending:
    idx = jnp.flip(idx, axis=-1)
  labels = jnp.take_along_axis(labels, idx, axis=-1)
  if label_mask is None:
    label_mask = True
  else:
    label_mask = jnp.take_along_axis(label_mask, idx, axis=-1)

  num_p = (labels > 0).sum(axis=-1, where=label_mask)
  num_p_above = jnp.cumsum((labels > 0) & label_mask, axis=-1)
  num_n = (labels == 0).sum(axis=-1, where=label_mask)
  num_n_above = jnp.cumsum((labels == 0) & label_mask, axis=-1)

  gmr = num_p_above.mean(axis=-1, where=(labels == 0) & label_mask) / num_p
  gmr_var = (num_n_above / num_n[:, None]).var(
      axis=-1, where=(labels > 0) & label_mask
  )
  return gmr, gmr_var


def least_squares_solve_mix(matrix, rhs, diag_loading=1e-3):
  # Assumes a real-valued matrix, with zero mean.
  adj_matrix = jnp.conjugate(jnp.swapaxes(matrix, -1, -2))
  cov_matrix = jnp.matmul(adj_matrix, matrix)

  # diag_loading ensures invertibility of the (pos. semi-definite) cov_matrix.
  cov_matrix += diag_loading * jnp.eye(
      cov_matrix.shape[-1], dtype=cov_matrix.dtype
  )
  return scipy.linalg.solve(
      cov_matrix, jnp.matmul(adj_matrix, rhs), assume_a='pos'
  )


def least_squares_mixit(reference, estimate):
  """Applies loss_fn after finding the best fit MixIt assignment."""
  # Reference shape is [B, M, T]
  # Estimate shape is [B, C, T]
  mix_matrix = least_squares_solve_mix(
      jnp.swapaxes(estimate, -1, -2), jnp.swapaxes(reference, -1, -2)
  )
  mix_matrix = jnp.swapaxes(mix_matrix, -1, -2)
  max_per_column = jnp.max(mix_matrix, axis=-2, keepdims=True)
  mix_matrix = jnp.where(mix_matrix == max_per_column, 1.0, 0.0)
  best_mix = mix_matrix @ estimate
  return best_mix, mix_matrix
