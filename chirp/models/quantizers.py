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
from typing import List
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class QuantizerOutputs:
  quantized: jnp.ndarray
  quantization_loss: jnp.ndarray
  nn_idx: jnp.ndarray
  codebook: jnp.ndarray


class BaseQuantizer(nn.Module):
  """Quantizer that can be used as a building block for ProductQuantizer.

  Attributes:
    num_centroids: The number of centroids.
    stop_gradient_codes: Whether to apply a stop gradient on the quantizer's
      codes, to protect them from being modified by downstream losses. Should
      always be True, and a future CL with remove this from being an option
      (keeping only for comparisons purposes with previous code).
  """
  num_centroids: int
  stop_gradient_codes: bool = True

  def get_num_centroids(self):
    return self.num_centroids

  def get_num_sections(self):
    return 1


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

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook = jax.lax.stop_gradient(codebook)

    return QuantizerOutputs(quantized, quantization_loss, nn_idx, codebook)


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

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook = jax.lax.stop_gradient(codebook)

    return QuantizerOutputs(quantized, quantization_loss, nn_idx, codebook)


class ProductQuantizer(nn.Module):
  """Product Quantizer.

  Attributes:
    num_sections: The number of sections to quantize.
    base_quantizer: A list of `num_sections` BaseQuantizer modules.
    stop_gradient_codes: Whether to apply a stop gradient on the quantizer's
      codes, to protect them from being modified by downstream losses. Should
      always be True, and a future CL with remove this from being an option
      (keeping only for comparisons purposes with previous code).
  """
  num_sections: int
  base_quantizer: List[nn.Module]
  stop_gradient_codes: bool = True

  def get_num_centroids(self):
    nc = [q.num_centroids for q in self.base_quantizer]
    assert len(
        list(set(nc))
    ) == 1, "Expected all base quantizers to have the same number of centroids."
    return nc[0]

  def get_num_sections(self):
    return self.num_sections

  @nn.compact
  def __call__(self, inputs):
    ns = self.num_sections

    # Divide the input into `num_sections` parts and quantize each separately.
    input_sections = jnp.split(inputs, ns, axis=-1)
    loss, quantized, nn_idx, codebook_list = [], [], [], []
    for quantizer, sec in zip(self.base_quantizer, input_sections):
      # Let `csz` denote the number of channels of `inputs` and `...` denotes
      # irrelevant dimensions like batch size and / or number of frames. Then:
      # outputs.quantized: [..., csz / ns].
      # outputs.nn_idx: [1, ...].
      # outputs.codebook: [1, nc, csz / ns].
      outputs = quantizer(sec)
      quantized.append(outputs.quantized)
      nn_idx.append(outputs.nn_idx)
      codebook_list.append(outputs.codebook)
      loss.append(outputs.quantization_loss)

    # Aggregate across 'sections' to get the following shapes:
    # quantized: [..., csz].
    # nn_idx: [ns, ...].
    # codebook: [ns, nc, csz / ns].
    quantized = jnp.concatenate(quantized, axis=-1)
    nn_idx = jnp.concatenate(nn_idx, axis=0)
    codebook = jnp.concatenate(codebook_list, axis=0)
    quantization_loss = jnp.mean(jnp.stack(loss, axis=0), axis=0)

    if self.stop_gradient_codes:
      codebook = jax.lax.stop_gradient(codebook)

    return QuantizerOutputs(quantized, quantization_loss, nn_idx, codebook)
