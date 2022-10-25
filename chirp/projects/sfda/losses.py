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

"""A set of common losses used by several SFDA methods."""

from typing import Optional

import flax
import jax
import jax.numpy as jnp


def label_xent(probabilities: jnp.ndarray,
               label: jnp.ndarray,
               eps: float = 1e-10,
               **_) -> jnp.ndarray:
  """Cross entropy for single-label classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes].
    label: One-hot labels, expected shape [*, num_classes].
    eps: For numerical stability

  Returns:
    Multi-class xent. Shape [*,].
  """
  xent = -(label * jnp.log(probabilities + eps)).sum(-1)
  return xent


def label_ent(probabilities: jnp.ndarray,
              eps: float = 1e-10,
              **_) -> jnp.ndarray:
  """Standard entropy used for single-label classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes]
    eps: For numerical stability.

  Returns:
    The entropy of probabilities, shape [*,]
  """
  return -(probabilities * jnp.log(probabilities + eps)).sum(-1)


def label_binary_ent(probabilities: jnp.ndarray,
                     label_mask: Optional[jnp.ndarray] = None,
                     eps: float = 1e-10,
                     **_) -> jnp.ndarray:
  """Computes averaged classwise binary entropy.

  Args:
    probabilities: Probabilities used to compute the binary entropies. Expected
      shape [*, num_classes].
    label_mask: Used to mask classes before averaging across classes. Expected
      shape [*, num_classes].
    eps: For numerical stability.

  Returns:
    The binary entropies, averaged across classes shape [*,]
  """
  if label_mask is None:
    label_mask = jnp.ones_like(probabilities)
  assert probabilities.shape == label_mask.shape, (probabilities.shape,
                                                   label_mask.shape)
  binary_entropies = -(probabilities * jnp.log(probabilities + eps) +
                       (1 - probabilities) * jnp.log((1 - probabilities) + eps)
                      )  # [..., num_classes]
  return (label_mask * binary_entropies).sum(axis=-1) / label_mask.sum(axis=-1)


def label_binary_xent(probabilities: jnp.ndarray,
                      label: jnp.ndarray,
                      label_mask: Optional[jnp.ndarray] = None,
                      eps: float = 1e-10,
                      **_) -> jnp.ndarray:
  """Computes averaged classwise binary cross-entropy.

  Args:
    probabilities: Shape [*, num_classes]
    label: Shape [*, num_classes]
    label_mask: Shape [*, num_classes]
    eps: For numerical stability.

  Returns:
    Average of per-class binary xent. Shape [*]
  """
  if label_mask is None:
    label_mask = jnp.ones_like(probabilities)
  assert probabilities.shape == label_mask.shape == label_mask.shape, (
      probabilities.shape, label_mask.shape, label_mask.shape)
  binary_entropies = -(label * jnp.log(probabilities + eps) +
                       (1 - label) * jnp.log((1 - probabilities) + eps)
                      )  # [..., num_classes]
  return (label_mask * binary_entropies).sum(axis=-1) / label_mask.sum(axis=-1)


def l2_loss(params: flax.core.scope.VariableDict):
  """Used to simulate weight decay."""
  loss = 0.
  for p in jax.tree_util.tree_leaves(params):
    loss += (p**2).sum()
  return loss
