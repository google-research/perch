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

"""Quantizers."""
from flax import linen as nn
import jax
from jax import numpy as jnp


class VectorQuantizer(nn.Module):
  """Vector Quantizer using L2-loss."""
  num_centroids: int

  def loss(self, inputs, quantized):
    return jnp.square(quantized - jax.lax.stop_gradient(inputs))

  @nn.compact
  def __call__(self, inputs):
    codebook = self.param("codebook", nn.initializers.normal(1.0),
                          (self.num_centroids, inputs.shape[-1]))

    # Expand codebook and feature dimensions for broadcasting.
    codes = jnp.expand_dims(codebook, range(inputs.ndim - 1))
    features = jax.lax.stop_gradient(inputs)
    features = jnp.expand_dims(features, -2)

    # Compute pairwise distances from features to codes.
    deltas = jnp.sum(jnp.square(codes - features), axis=-1)

    # Find nearest neighbor indices.
    nn_idx = jnp.argmin(deltas, axis=-1)
    encoding = jax.nn.one_hot(nn_idx, self.num_centroids)
    quantized = jnp.matmul(encoding, codes)

    quantization_loss = self.loss(inputs, quantized)
    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    return quantized, quantization_loss, nn_idx, codebook


class VectorQuantizerEnt(nn.Module):
  """Vector Quantizer using entropy loss."""
  num_centroids: int
  gamma: float = 1.0

  def loss(self, gamma, scores):
    scores_shape = scores.shape
    scores = jnp.reshape(scores, [-1, scores.shape[-1]])
    h_clust = jnp.sum(scores * jnp.log2(scores + 1e-8), axis=-1)
    h_clust = -jnp.mean(h_clust)

    diversity = jnp.mean(scores, axis=0)
    h_diversity = -jnp.sum(diversity * jnp.log2(diversity + 1e-8))
    loss = h_clust - gamma * h_diversity
    loss = jnp.full(scores_shape, loss)
    return loss

  @nn.compact
  def __call__(self, inputs):
    codebook = self.param("codebook", nn.initializers.normal(1.0),
                          (self.num_centroids, inputs.shape[-1]))

    # Expand codebook and feature dimensions for broadcasting.
    codes = jnp.expand_dims(codebook, range(inputs.ndim - 1))
    features = inputs
    features = jax.lax.stop_gradient(features)
    features = jnp.expand_dims(features, -2)
    similarity = jnp.sum(features * codes, axis=-1)
    scores = jax.nn.softmax(similarity, axis=-1)

    # Find nearest neighbor indices.
    nn_idx = jnp.argmax(scores, axis=-1)
    encoding = jax.nn.one_hot(nn_idx, self.num_centroids)
    quantized = jnp.matmul(encoding, codes)
    quantized -= jnp.mean(quantized, axis=-1, keepdims=True)
    quantized /= jnp.linalg.norm(quantized, axis=-1, keepdims=True)

    quantization_loss = self.loss(self.gamma, scores)
    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    return quantized, quantization_loss, nn_idx, codebook
