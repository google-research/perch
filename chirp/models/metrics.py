# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

from jax import lax
from jax import numpy as jnp
from jax import scipy
import optax


def mean_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  mean = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)
  return lax.pmean(mean, axis_name="batch")


def map_(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  return average_precision(scores=logits, labels=labels)


def cmap_(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  return average_precision(scores=logits.T, labels=labels.T)


def log_mse_loss(source: jnp.ndarray,
                 estimate: jnp.ndarray,
                 max_snr: float = 1e6,
                 eps=1e-8) -> jnp.ndarray:
  """Negative log MSE loss, the negated log of SNR denominator.

  With default max_snr = 1e6, this gives the usual log((source-estimate)**2).
  When a max_snr is specified, it acts as a soft threshold clamping the loss.

  Args:
    source: Groundtruth audio, with time in the last dimension.
    estimate: Estimate of Groundtruth with the same shape as source.
    max_snr: SNR threshold for minimal loss. The default 1e6 yields an unbiased
      log mse calculation.
    eps: Epsilon for log stabilization.

  Returns:
    Array of loss values.
  """
  err_pow = jnp.sum((source - estimate)**2, axis=-1)
  snrfactor = 10.**(-max_snr / 10.)
  ref_pow = jnp.sum(jnp.square(source), axis=-1)
  bias = snrfactor * ref_pow
  return 10 * jnp.log10(bias + err_pow + eps)


def negative_snr_loss(source: jnp.ndarray,
                      estimate: jnp.ndarray,
                      max_snr: float = 1e6,
                      eps: float = 1e-8) -> jnp.ndarray:
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
  snrfactor = 10.**(-max_snr / 10.)
  ref_pow = jnp.sum(jnp.square(source), axis=-1)
  bias = snrfactor * ref_pow

  numer = 10. * jnp.log10(ref_pow + eps)
  err_pow = jnp.sum(jnp.square(source - estimate), axis=-1)
  return 10 * jnp.log10(bias + err_pow + eps) - numer


def source_sparsity_l1l2ratio_loss(separated_waveforms: jnp.ndarray,
                                   mix_of_mix_waveforms: jnp.ndarray,
                                   eps: float = 1e-8) -> jnp.ndarray:
  """Sparsity loss for separated audio.

  Computes the ratio of L1 to L2 measures across source rms power.
  Note, this is actually the square root of the weighted mean when input
  and weights are rms power.
  See Section 2.3 in https://arxiv.org/abs/2106.00847

  Args:
    separated_waveforms: Estimated separated audio with shape [Batch, Channels,
      Time].
    mix_of_mix_waveforms: MoM audio, which separated_waveforms separates.
    eps: Epsilon for stability.

  Returns:
    Loss tensor.
  """
  src_pow = jnp.mean(jnp.square(separated_waveforms), axis=2)
  src_rms = jnp.sqrt(src_pow)

  l1norm = jnp.mean(src_rms, axis=1)
  mixture_pow = jnp.mean(jnp.square(mix_of_mix_waveforms), axis=2)
  l2norm_mixture = jnp.sqrt(mixture_pow)
  loss = l1norm / (l2norm_mixture + eps)
  return loss


def average_precision(scores: jnp.ndarray,
                      labels: jnp.ndarray,
                      interpolated: bool = False) -> jnp.ndarray:
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
    interpolated: Whether to use interpolation.

  Returns:
    The average precision.
  """
  idx = jnp.flip(jnp.argsort(scores), axis=-1)
  scores = jnp.take_along_axis(scores, idx, axis=-1)
  labels = jnp.take_along_axis(labels, idx, axis=-1)
  pr_curve = jnp.cumsum(labels, axis=-1) / (jnp.arange(labels.shape[-1]) + 1)
  if interpolated:
    pr_curve = lax.cummax(pr_curve, reverse=True, axis=-1)

  # In case of an empty row, assign precision = 1, and avoid dividing by zero.
  mask = jnp.float32(jnp.sum(labels, axis=-1) == 0)
  raw_av_prec = (
      jnp.sum(pr_curve * labels, axis=-1) /
      jnp.maximum(jnp.sum(labels, axis=-1), 1.0))
  return mask + (1 - mask) * raw_av_prec


def least_squares_solve_mix(matrix, rhs, diag_loading=1e-3):
  # Assumes a real-valued matrix, with zero mean.
  adj_matrix = jnp.conjugate(jnp.swapaxes(matrix, -1, -2))
  cov_matrix = jnp.matmul(adj_matrix, matrix)

  # diag_loading ensures invertibility of the (pos. semi-definite) cov_matrix.
  cov_matrix += diag_loading * jnp.eye(
      cov_matrix.shape[-1], dtype=cov_matrix.dtype)
  return scipy.linalg.solve(
      cov_matrix, jnp.matmul(adj_matrix, rhs), sym_pos=True)


def least_squares_mixit(reference, estimate):
  """Applies loss_fn after finding the best fit MixIt assignment."""
  # Reference shape is [B, M, T]
  # Estimate shape is [B, C, T]
  mix_matrix = least_squares_solve_mix(
      jnp.swapaxes(estimate, -1, -2), jnp.swapaxes(reference, -1, -2))
  mix_matrix = jnp.swapaxes(mix_matrix, -1, -2)
  max_per_column = jnp.max(mix_matrix, axis=-2, keepdims=True)
  mix_matrix = jnp.where(mix_matrix == max_per_column, 1.0, 0.0)
  best_mix = mix_matrix @ estimate
  return best_mix, mix_matrix
