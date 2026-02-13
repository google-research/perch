# coding=utf-8
# Copyright 2026 The Perch Authors.
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

"""Layers which reduce spatial embeddings to logits."""

from typing import Any, Protocol, Type

from flax import linen as nn
import jax
from jax import numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class ClassifierHead(Protocol):
  """Classifier head."""

  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    ...


class DenseHead(ClassifierHead, nn.Module):
  """Simple dense head, extracting logits from the average embedding."""

  num_classes: int
  dropout_rate: float = 0.0
  linear_projection: Type[nn.Module] = nn.Dense
  linear_projection_kwargs: dict[str, Any] | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    avg_embedding = jnp.mean(inputs, axis=(-2, -3))
    if train and self.dropout_rate > 0:
      avg_embedding = nn.Dropout(
          rate=self.dropout_rate, deterministic=not train
      )(avg_embedding)
    linear_projection_kwargs = self.linear_projection_kwargs or {}
    linear_projection = self.linear_projection(
        self.num_classes, **linear_projection_kwargs
    )
    return linear_projection(avg_embedding)


class LowRankHead(ClassifierHead, nn.Module):
  """Low rank head, extracting logits from the average embedding."""

  rank: int
  num_classes: int
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    avg_embedding = jnp.mean(inputs, axis=(-2, -3))
    if not train:
      return jnp.zeros(avg_embedding.shape[:-1] + (self.num_classes,))
    if self.dropout_rate > 0:
      avg_embedding = nn.Dropout(
          rate=self.dropout_rate, deterministic=not train
      )(avg_embedding)
    projection = nn.Dense(self.rank)(avg_embedding)
    return nn.Dense(self.num_classes, use_bias=self.use_bias)(projection)


class ProtoPNetHead(ClassifierHead, nn.Module):
  """ProtoPeanut head.

  See https://arxiv.org/pdf/2404.10420 for details.

  Attributes:
    num_prototypes: The number of prototypes per class to use.
    num_classes: The number of classes to predict.
    non_negative_kernel: Whether to enforce non-negative kernel values.
    ortho_loss_weight: The weight to apply to the orthogonality loss.
    kernel_init_value: The value to initialize the output kernel with.
    bias_init_value: The value to initialize the bias with.
    dtype: The dtype to use for the computation.
    eps: A small value to add to the denominator to avoid division by zero.
  """

  num_prototypes: int
  num_classes: int
  dropout_rate: float = 0.3
  non_negative_kernel: bool = True
  ortho_loss_weight: float = 1.0
  kernel_init_value: float = 2.0
  bias_init_value: float = -2.0
  dtype: jnp.dtype = jnp.float32
  eps: float = 1e-5

  def compute_ortho_loss(self, unit_kernel: Array) -> Array:
    """Computes orthogonality loss for the given kernel."""
    # proto_sim has shape [num_classes, num_prototypes, num_prototypes]
    proto_sim = jnp.matmul(unit_kernel.transpose(0, 2, 1), unit_kernel)
    proto_sim -= jnp.eye(self.num_prototypes)[jnp.newaxis, :, :]
    ortho_loss = (
        self.ortho_loss_weight
        * (proto_sim**2).sum()
        / (self.num_prototypes**2 * self.num_classes)
    )
    self.sow('intermediates', 'ortho_loss', ortho_loss)
    return ortho_loss

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """ProtoPNet classification head."""
    embedding_dim = inputs.shape[-1]
    kernel = self.param(
        'prototypes',
        nn.initializers.truncated_normal(
            lower=1e-5, upper=2.0, dtype=self.dtype
        ),
        [self.num_classes, embedding_dim, self.num_prototypes],
        self.dtype,
    )
    unit_kernel = kernel / (
        jnp.linalg.norm(kernel, axis=1, keepdims=True) + self.eps
    )
    self.compute_ortho_loss(unit_kernel)
    unit_inputs = inputs / (
        jnp.linalg.norm(inputs, axis=-1, keepdims=True) + self.eps
    )
    sims = jnp.dot(unit_inputs, unit_kernel)
    # Reduce over the spatial axes to [B, num_classes, num_prototypes].
    sims = jnp.max(sims, axis=(-4, -3))

    # Compute class predictions from prototypes.
    protop_kernel = self.param(
        'protop_kernel',
        nn.initializers.constant(self.kernel_init_value),
        [self.num_classes, self.num_prototypes],
        self.dtype,
    )
    bias = self.param(
        'bias',
        nn.initializers.constant(self.bias_init_value),
        [self.num_classes],
        self.dtype,
    )
    if self.non_negative_kernel:
      protop_kernel = jnp.maximum(protop_kernel, 0)
    scores = jnp.sum(sims * protop_kernel, axis=-1) + bias
    return scores
