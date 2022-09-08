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


class BaseQuantizer(nn.Module):
  """Quantizer that can be used as a building block for ProductQuantizer."""
  num_centroids: int

  def get_num_centroids(self):
    return self.num_centroids


class VectorQuantizer(BaseQuantizer):
  """Vector Quantizer using L2-loss."""

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


class VectorQuantizerEnt(BaseQuantizer):
  """Vector Quantizer using entropy loss."""
  gamma: float = 1.0

  def loss(self, scores):
    scores_shape = scores.shape
    scores = jnp.reshape(scores, [-1, scores.shape[-1]])
    h_clust = jnp.sum(scores * jnp.log2(scores + 1e-8), axis=-1)
    h_clust = -jnp.mean(h_clust)

    diversity = jnp.mean(scores, axis=0)
    h_diversity = -jnp.sum(diversity * jnp.log2(diversity + 1e-8))
    loss = h_clust - self.gamma * h_diversity
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

    quantization_loss = self.loss(scores)
    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    return quantized, quantization_loss, nn_idx, codebook


class ProductQuantizer(nn.Module):
  """Vector Quantizer."""
  num_sections: int
  base_quantizer: nn.Module

  def get_num_centroids(self):
    return self.base_quantizer.num_centroids

  @nn.compact
  def __call__(self, inputs):
    ns = self.num_sections

    # Divide the input into `num_sections` parts and quantize each separately.
    input_sections = jnp.split(inputs, ns, axis=-1)
    quantized, nn_idx, codebook_list = [], [], []
    for sec in input_sections:
      # Let `csz` denote the number of channels of `inputs` and `...` denotes
      # irrelevant dimensions like batch size and / or number of frames. Then:
      # quantized_sec: [..., csz / ns].
      # nn_idx_sec: [...].
      # codebook_sec: [nc, csz / ns].
      quantized_sec, _, nn_idx_sec, codebook_sec = self.base_quantizer(sec)
      quantized.append(quantized_sec)
      nn_idx.append(nn_idx_sec)
      codebook_list.append(codebook_sec)

    # Aggregate across 'sections' to get the following shapes:
    # quantized: [..., csz].
    # nn_idx: [ns, ...].
    # codebook: [ns, nc, csz / ns].
    quantized = jnp.concatenate(quantized, axis=-1)
    nn_idx = jnp.stack(nn_idx, axis=0)
    codebook = jnp.stack(codebook_list, axis=0)

    if isinstance(self.base_quantizer, VectorQuantizer):
      quantization_loss = self.base_quantizer.loss(inputs, quantized)
    elif isinstance(self.base_quantizer, VectorQuantizerEnt):
      # [..., ns, csz/ns].
      input_sections = jnp.stack(input_sections, axis=-2)
      features = jax.lax.stop_gradient(input_sections)
      # [..., ns, 1, csz/ns].
      features = jnp.expand_dims(features, -2)
      # [..., ns, nc, csz/ns].
      codes = jnp.expand_dims(codebook, range(input_sections.ndim - 2))
      # [..., ns, nc].
      similarity = jnp.sum(features * codes, axis=-1)
      scores = jax.nn.softmax(similarity, axis=-1)
      # Bring ns to be the first dimension.
      offset = len(scores.shape[:-2])
      scores = jnp.transpose(scores,
                             [offset] + list(range(offset)) + [offset + 1])
      quantization_loss = self.base_quantizer.loss(scores)
    else:
      raise ValueError(
          "`base_quantizer` was expected to be VectorQuantizer or VectorQuantizerEnt"
      )
    return quantized, quantization_loss, nn_idx, codebook
