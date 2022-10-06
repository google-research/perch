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

import flax
import jax
import jax.numpy as jnp
import optax


def label_binary_ent(probas: jnp.ndarray,
                     label_mask: jnp.ndarray,
                     eps: float = 1e-10,
                     **_) -> jnp.ndarray:
  """Computes averaged classwise binary entropy.

  Args:
    probas: Probabilities used to compute the binary entropies. Expected shape
      [*, num_classes].
    label_mask: Used to mask classes before averaging across classes. Expected
      shape [*, num_classes].
    eps: For numerical stability.

  Returns:
    The binary entropies, averaged across classes shape [*,]
  """
  assert probas.shape == label_mask.shape, (probas.shape, label_mask.shape)
  binary_entropies = -(probas * jnp.log(probas + eps) +
                       (1 - probas) * jnp.log((1 - probas) + eps)
                      )  # [..., num_classes]
  return (label_mask * binary_entropies).sum(axis=-1) / label_mask.sum(axis=-1)


def label_binary_xent(logits: jnp.ndarray, label: jnp.ndarray, label_mask,
                      **_) -> jnp.ndarray:
  """Computes averaged classwise binary cross-entropy.

  Args:
    logits: Shape [*, num_classes]
    label: Shape [*, num_classes]
    label_mask: Shape [*, num_classes]

  Returns:
    Average of per-class binary xent. Shape [*]
  """
  assert logits.shape == label_mask.shape == label_mask.shape, (
      logits.shape, label_mask.shape, label_mask.shape)
  cross_entropy = optax.sigmoid_binary_cross_entropy(logits, label)
  return jnp.sum(label_mask * cross_entropy, axis=-1) / label_mask.sum(axis=-1)


def l2_loss(params: flax.core.scope.VariableDict):
  """Used to simulate weight decay."""
  loss = 0.
  for p in jax.tree_util.tree_leaves(params):
    loss += (p**2).sum()
  return loss
