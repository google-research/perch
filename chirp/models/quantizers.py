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
from typing import List, Optional
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
  cluster_counts: List[jnp.ndarray]


class BaseQuantizer(nn.Module):
  """Quantizer that can be used as a building block for ProductQuantizer.

  Attributes:
    num_centroids: The number of centroids.
    stop_gradient_codes: Whether to apply a stop gradient on the quantizer's
      codes, to protect them from being modified by downstream losses. Should
      always be True, and a future CL with remove this from being an option
      (keeping only for comparisons purposes with previous code).
    cross_replica_axis: Name of the cross-replica axis for applying ops
      requiring cross-replica reductions.
    ema_decay: Decay rate for EMA operations.
    init_scale: Scale for codebook initialization.
  """
  num_centroids: int
  stop_gradient_codes: bool = True
  cross_replica_axis: Optional[str] = None
  ema_decay: float = 0.99
  init_scale: float = 0.1

  def get_num_centroids(self):
    return self.num_centroids

  def get_num_sections(self):
    return 1

  def create_codebook(self, flat_inputs):
    """Default codebook variable."""
    embedding_dim = flat_inputs.shape[-1]
    init_fn = jax.nn.initializers.variance_scaling(self.init_scale, 'fan_avg',
                                                   'normal')
    codebook = self.param('codebook', init_fn,
                          (self.num_centroids, embedding_dim))
    return codebook

  def update_cluster_counts(self, encodings, train):
    """Track cluster utilization with an EMA counter."""
    counts = jnp.sum(encodings, axis=range(len(encodings.shape) - 1))
    cluster_counts = self.variable('quantizer', 'cluster_counts', jnp.ones,
                                   [self.num_centroids])
    if not train:
      # TODO(tomdenton): Define some better behavior for eval?
      # Would be nice to re-init during eval to get eval-specific metrics.
      return cluster_counts.value
    self._ema_update(cluster_counts, counts)
    return cluster_counts.value

  def _ema_update(self, variable, new_value):
    """Apply an EMA variable update, possibly in cross-device context."""
    if self.cross_replica_axis:
      new_value = jax.lax.psum(new_value, axis_name=self.cross_replica_axis)
    variable.value = (
        self.ema_decay * variable.value + (1.0 - self.ema_decay) * new_value)


class VectorQuantizer(BaseQuantizer):
  """Vector Quantizer using L2-loss.

  Attributes:
    commitment_loss: Loss weight for propagating quantization loss to inputs.
  """
  commitment_loss: float = 0.0

  def loss(self, inputs, quantized):
    quant_loss = jnp.square(quantized - jax.lax.stop_gradient(inputs))
    if self.commitment_loss > 0:
      encoder_loss = jnp.square(jax.lax.stop_gradient(quantized) - inputs)
      quant_loss += self.commitment_loss * encoder_loss
    return quant_loss

  @nn.compact
  def __call__(self, inputs, train):
    embedding_dim = inputs.shape[-1]
    flat_inputs = jnp.reshape(inputs, [-1, embedding_dim])
    codebook = self.create_codebook(flat_inputs)

    # Find nearest neighbor indices.
    distances = (
        jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
        2 * jnp.matmul(flat_inputs, codebook.T) +
        jnp.sum(jnp.square(codebook.T), 0, keepdims=True))
    nn_idx = jnp.argmin(distances, axis=1)
    encodings = jax.nn.one_hot(nn_idx, self.num_centroids)
    counts = self.update_cluster_counts(encodings, train)
    quantized = jnp.matmul(encodings, codebook)
    quantized = jnp.reshape(quantized, inputs.shape)
    nn_idx = jnp.reshape(nn_idx, inputs.shape[:-1])
    quantization_loss = self.loss(inputs, quantized)

    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook_values = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook_values = jax.lax.stop_gradient(codebook_values)

    return QuantizerOutputs(quantized, quantization_loss, nn_idx,
                            codebook_values, [counts])


class VectorQuantizerEnt(BaseQuantizer):
  """Vector Quantizer using entropy loss."""
  gamma: float = 1.0

  def loss(self, scores):
    scores = jnp.reshape(scores, [-1, scores.shape[-1]])
    h_clust = jnp.sum(scores * jnp.log2(scores + 1e-8), axis=-1)
    h_clust = -jnp.mean(h_clust)

    diversity = jnp.mean(scores, axis=0)
    h_diversity = -jnp.sum(diversity * jnp.log2(diversity + 1e-8))
    loss = h_clust - self.gamma * h_diversity
    return loss

  @nn.compact
  def __call__(self, inputs, train):
    embedding_dim = inputs.shape[-1]
    flat_inputs = jnp.reshape(inputs, [-1, embedding_dim])
    codebook = self.create_codebook(flat_inputs)

    # Expand codebook and feature dimensions for broadcasting.
    codes = jnp.expand_dims(codebook, range(flat_inputs.ndim - 1))
    features = jax.lax.stop_gradient(flat_inputs)
    features = jnp.expand_dims(features, -2)
    similarity = jnp.sum(features * codes, axis=-1)
    scores = jax.nn.softmax(similarity, axis=-1)

    # Find nearest neighbor indices.
    nn_idx = jnp.argmax(scores, axis=-1)
    encodings = jax.nn.one_hot(nn_idx, self.num_centroids)
    counts = self.update_cluster_counts(encodings, train)
    quantized = jnp.matmul(encodings, codes)
    quantized -= jnp.mean(quantized, axis=-1, keepdims=True)
    quantized /= jnp.linalg.norm(quantized, axis=-1, keepdims=True)
    quantized = jnp.reshape(quantized, inputs.shape)
    nn_idx = jnp.reshape(nn_idx, inputs.shape[:-1])

    quantization_loss = self.loss(scores)
    quantization_loss = jnp.full(inputs.shape[:-1] + (self.num_centroids,),
                                 quantization_loss)
    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook_values = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook_values = jax.lax.stop_gradient(codebook_values)

    return QuantizerOutputs(quantized, quantization_loss, nn_idx,
                            codebook_values, [counts])


class ProductQuantizer(nn.Module):
  """Product Quantizer.

  Attributes:
    num_sections: The number of sections to quantize.
    base_quantizers: A list of `num_sections` BaseQuantizer modules.
    stop_gradient_codes: Whether to apply a stop gradient on the quantizer's
      codes, to protect them from being modified by downstream losses. Should
      always be True, and a future CL with remove this from being an option
      (keeping only for comparisons purposes with previous code).
  """
  num_sections: int
  base_quantizers: List[BaseQuantizer]
  stop_gradient_codes: bool = True

  def get_num_centroids(self):
    nc = [q.num_centroids for q in self.base_quantizers]
    assert len(
        list(set(nc))
    ) == 1, 'Expected all quantizers to have the same number of centroids.'
    return nc[0]

  def get_num_sections(self):
    return self.num_sections

  @nn.compact
  def __call__(self, inputs, train):
    ns = self.num_sections

    # Divide the input into `num_sections` parts and quantize each separately.
    input_sections = jnp.split(inputs, ns, axis=-1)
    loss, quantized, nn_idx, codebook_list, counts = [], [], [], [], []
    for quantizer, sec in zip(self.base_quantizers, input_sections):
      # Let `csz` denote the number of channels of `inputs` and `...` denotes
      # irrelevant dimensions like batch size and / or number of frames. Then:
      # outputs.quantized: [..., csz / ns].
      # outputs.nn_idx: [1, ...].
      # outputs.codebook: [1, nc, csz / ns].
      outputs = quantizer(sec, train)
      quantized.append(outputs.quantized)
      nn_idx.append(outputs.nn_idx)
      codebook_list.append(outputs.codebook)
      loss.append(outputs.quantization_loss)
      counts += outputs.cluster_counts

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

    return QuantizerOutputs(quantized, quantization_loss, nn_idx, codebook,
                            counts)
