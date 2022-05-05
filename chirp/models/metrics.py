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


def average_precision(scores: jnp.ndarray,
                      labels: jnp.ndarray,
                      interpolated: bool = False,
                      axis: int = -1) -> jnp.ndarray:
  """Average precision.

  The average precision is the area under the precision-recall curve. When
  using interpolation we take the maximum precision over all smaller recalls.
  The intuition is that it often makes sense to evaluate more documents if the
  total percentage of relevant documents increases.

  Args:
    scores: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    interpolated: Whether to use interpolation.
    axis: The axis containing the scores and class labels.

  Returns:
    The average precision.
  """
  idx = jnp.flip(jnp.argsort(scores), axis)
  scores = jnp.take_along_axis(scores, idx, axis=axis)
  labels = jnp.take_along_axis(labels, idx, axis=axis)
  pr_curve = jnp.cumsum(
      labels, axis=axis) / (
          jnp.arange(labels.shape[axis]) + 1)
  if interpolated:
    pr_curve = lax.cummax(pr_curve, reverse=True, axis=axis)
  return jnp.sum(pr_curve * labels, axis=axis) / jnp.sum(labels, axis=axis)
