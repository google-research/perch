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

import enum


import flax
import jax
import jax.numpy as jnp


class ReduceStrategy(enum.Enum):
  """Strategies to reduce an axis.

  Attributes:
      NONE: No reduction
      AVERAGE: Average reduction
  """

  NONE = "none"
  AVERAGE = "average"


def label_kl(
    probabilities: jnp.ndarray,
    label: jnp.ndarray,
    label_mask: jnp.ndarray | None,
    eps: float = 1e-10, **_) -> jnp.ndarray:
  """Kulback-Leibler divergence for single-class classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes].
    label: One-hot labels, expected shape [*, num_classes].
    label_mask: Array representing classes to be kept, shape [*, num_classes].
    eps: For numerical stability

  Returns:
    Single-class Kulback-Leibler divergence between probabilities and label.
    Shape [*,].
  """
  return label_xent(
      probabilities, label, label_mask, eps) - label_ent(
      probabilities, label_mask, eps)  # pytype: disable=wrong-arg-types  # jax-ndarray


def label_binary_kl(
    probabilities: jnp.ndarray, label: jnp.ndarray, **_
) -> jnp.ndarray:
  """Kulback-Leibler divergence for multi-class classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes].
    label: One-hot labels, expected shape [*, num_classes].

  Returns:
    Multi-class Kulback-Leibler divergence between probabilities and label.
    Shape [*, num_classes].
  """
  return label_binary_xent(
      probabilities, label, class_reduce=ReduceStrategy.NONE
  ) - label_binary_ent(probabilities, class_reduce=ReduceStrategy.NONE)


def label_xent(
    probabilities: jnp.ndarray,
    label: jnp.ndarray,
    label_mask: jnp.ndarray | None,
    sample_mask: jnp.ndarray | None = None,
    eps: float = 1e-10,
    **_,
) -> jnp.ndarray:
  """Cross entropy for single-label classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes].
    label: One-hot labels, expected shape [*, num_classes].
    label_mask: label_mask: Array representing classes to be kept,
      shape [*, num_classes].
    sample_mask: A way to mask out some samples when computing the loss. Useful
      for instance to only keep high-confidence samples in pseudo-labelling.
    eps: For numerical stability

  Returns:
    Multi-class xent. Shape [*,].
  """
  if sample_mask is None:
    sample_mask = jnp.ones(probabilities.shape[:-1])
  if label_mask is not None and (
          label_mask.shape[-1] == probabilities.shape[-1]):
    xent = -((label * jnp.log(probabilities + eps)) * label_mask).sum(-1)
  else:
    # TODO(mboudiaf) If label_mask is not None, check that probabilities are
    # already masked. In other words, ensure
    # probabilities.shape[-1] == label_mask.sum(-1)
    xent = -(label * jnp.log(probabilities + eps)).sum(-1)
  return sample_mask * xent


def label_ent(probabilities: jnp.ndarray,
              label_mask: jnp.ndarray | None,
              eps: float = 1e-10,
              **_) -> jnp.ndarray:
  """Standard entropy used for single-label classification settings.

  Args:
    probabilities: Model's probabilities, expected shape [*, num_classes]
    label_mask: label_mask: Array representing classes to be kept,
      shape [*, num_classes].
    eps: For numerical stability.

  Returns:
    The entropy of probabilities, shape [*,]
  """
  if label_mask is not None and label_mask.shape[-1] == probabilities.shape[-1]:
    ent = -((probabilities * jnp.log(probabilities + eps)) * label_mask).sum(-1)
  else:
    # TODO(mboudiaf) If label_mask is not None, check that probabilities are
    # already masked. In other words, ensure
    # probabilities.shape[-1] == label_mask.sum(-1)
    ent = -((probabilities * jnp.log(probabilities + eps))).sum(-1)
  return ent

def label_binary_ent(
    probabilities: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    eps: float = 1e-10,
    class_reduce: ReduceStrategy = ReduceStrategy.AVERAGE,
    **_,
) -> jnp.ndarray:
  """Computes averaged classwise binary entropy.

  Args:
    probabilities: Probabilities used to compute the binary entropies. Expected
      shape [*, num_classes].
    label_mask: Used to mask classes before averaging across classes. Expected
      shape [*, num_classes].
    eps: For numerical stability.
    class_reduce: Class reduction strategy.

  Returns:
    The binary entropies, averaged across classes shape [*,]

  Raises:
    ValueError: In case class_reduce is not a known ReduceStrategy.
  """
  if label_mask is None:
    label_mask = jnp.ones_like(probabilities)
  assert probabilities.shape == label_mask.shape, (
      probabilities.shape,
      label_mask.shape,
  )
  binary_entropies = -(
      probabilities * jnp.log(probabilities + eps)
      + (1 - probabilities) * jnp.log((1 - probabilities) + eps)
  )  # [..., num_classes]
  if class_reduce == ReduceStrategy.AVERAGE:
    return (label_mask * binary_entropies).sum(axis=-1) / (
        label_mask.sum(axis=-1) + eps
    )
  elif class_reduce == ReduceStrategy.NONE:
    return label_mask * binary_entropies
  else:
    raise ValueError(f"Unknown reduce strategy {class_reduce} used.")


def label_binary_xent(
    probabilities: jnp.ndarray,
    label: jnp.ndarray,
    label_mask: jnp.ndarray | None = None,
    eps: float = 1e-10,
    class_reduce: ReduceStrategy = ReduceStrategy.AVERAGE,
    **_,
) -> jnp.ndarray:
  """Computes averaged classwise binary cross-entropy.

  Args:
    probabilities: Shape [*, num_classes]
    label: Shape [*, num_classes]
    label_mask: Shape [*, num_classes]
    eps: For numerical stability.
    class_reduce: Class reduction strategy.

  Returns:
    Average of per-class binary xent. Shape [*]

  Raises:
    ValueError: In case class_reduce is not a known ReduceStrategy.
  """
  if label_mask is None:
    label_mask = jnp.ones_like(probabilities)
  assert probabilities.shape == label_mask.shape == label_mask.shape, (
      probabilities.shape,
      label_mask.shape,
      label_mask.shape,
  )
  binary_entropies = -(
      label * jnp.log(probabilities + eps)
      + (1 - label) * jnp.log((1 - probabilities) + eps)
  )  # [..., num_classes]
  if class_reduce == ReduceStrategy.AVERAGE:
    return (label_mask * binary_entropies).sum(axis=-1) / (
        label_mask.sum(axis=-1) + eps
    )
  elif class_reduce == ReduceStrategy.NONE:
    return label_mask * binary_entropies
  else:
    raise ValueError(f"Unknown reduce strategy {class_reduce} used.")


def l2_loss(params: flax.core.scope.VariableDict):
  """Used to simulate weight decay."""
  loss = 0.0
  for p in jax.tree_util.tree_leaves(params):
    loss += (p**2).sum()
  return loss
