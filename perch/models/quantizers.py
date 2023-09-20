# coding=utf-8
# Copyright 2023 The Perch Authors.
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
import enum
from typing import Sequence
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
  cluster_counts: list[jnp.ndarray]


class QuantizationStrategy(enum.Enum):
  """The Quantization strategy."""

  PRODUCT_QUANTIZATION = 'product_quantization'
  RESIDUAL_QUANTIZATION = 'residual_quantization'


def refresh_codebooks(
    model_params: flax.core.FrozenDict,
    model_state: flax.core.FrozenDict,
    rng: jnp.ndarray,
    utilization_thresh: float,
    init_scalar: float = 0.1,
) -> tuple[flax.core.FrozenDict, flax.core.FrozenDict]:
  """Restart dead codebook vectors.

  When usage falls below the target utilization_thresh, codebook entries are
  re-initialized by adding noise to the most-used codebook entry.

  Args:
    model_params: Params tree containing codebooks.
    model_state: State tree containing codebook usage counts.
    rng: RNG used to generate re-initialization noise.
    utilization_thresh: Threshold for restarting a codebook entry. Note that
      this is expressed as a proportion of the uniform probability. (ie, the
      actual threshold is utilization_thresh/num_centroids.)
    init_scalar: Scalar for generated initialization noise.

  Returns:
    Updated model_params and model_state.
  """
  flat_params = flax.traverse_util.flatten_dict(model_params)
  flat_model_state = flax.traverse_util.flatten_dict(model_state)

  for k, codebook in flat_params.items():
    # Check that the lowest variable name is codebook; ignore all other params.
    if k[-1] != 'codebook':
      continue
    # Get the corresponding codebook assignment counts.
    # These counts are generated under the 'quantizer' collection.
    count_key = ('quantizer',) + k[:-1] + ('cluster_counts',)
    counts = flat_model_state[count_key]
    num_centroids = counts.shape[0]
    cl_probs = flat_model_state[count_key] / num_centroids
    thresh = utilization_thresh / num_centroids
    replace = (cl_probs < thresh)[:, jnp.newaxis]

    # To get replacement entries, take existing codebook entries according to
    # their popularity and add a bit of noise.
    noise_key, rng = jax.random.split(rng)
    init_fn = jax.nn.initializers.variance_scaling(
        init_scalar, 'fan_avg', 'normal', dtype=codebook.dtype
    )
    init_noise = init_fn(noise_key, codebook.shape)

    categorical_key, rng = jax.random.split(rng)
    idxs = jax.random.categorical(
        categorical_key, counts, shape=[num_centroids]
    )
    replacement_entries = codebook[idxs, :]

    init_values = replacement_entries + init_noise
    updated_codebook = replace * init_values + (1.0 - replace) * codebook
    updated_counts = (
        replace[:, 0] * jnp.ones_like(counts) + (1.0 - replace[:, 0]) * counts
    )

    flat_params[k] = updated_codebook
    flat_model_state[count_key] = updated_counts

  unflat_params = flax.traverse_util.unflatten_dict(flat_params)
  unflat_params = flax.core.frozen_dict.freeze(unflat_params)
  unflat_model_state = flax.traverse_util.unflatten_dict(flat_model_state)
  unflat_model_state = flax.core.frozen_dict.freeze(unflat_model_state)
  return unflat_params, unflat_model_state


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
  cross_replica_axis: str | None = None
  ema_decay: float = 0.99
  init_scale: float = 0.1

  def get_num_centroids(self):
    return self.num_centroids

  def get_num_sections(self):
    return 1

  def create_codebook(self, flat_inputs):
    """Default codebook variable."""
    embedding_dim = flat_inputs.shape[-1]
    init_fn = jax.nn.initializers.variance_scaling(
        self.init_scale, 'fan_avg', 'normal'
    )
    codebook = self.param(
        'codebook', init_fn, (self.num_centroids, embedding_dim)
    )
    return codebook

  def update_cluster_counts(self, encodings, train):
    """Track cluster utilization with an EMA counter."""
    counts = jnp.sum(encodings, axis=range(len(encodings.shape) - 1))
    cluster_counts = self.variable(
        'quantizer', 'cluster_counts', jnp.ones, [self.num_centroids]
    )
    if not train:
      # TODO(tomdenton): Define some better behavior for eval?
      # Would be nice to re-init during eval to get eval-specific metrics.
      return cluster_counts.value
    self._ema_update(cluster_counts, counts)
    return cluster_counts.value

  def update_mean_estimate(self, flat_inputs, train):
    """Update an EMA estimate of the feature means."""
    embedding_dim = flat_inputs.shape[-1]
    feature_means = self.variable(
        'quantizer', 'feature_means', jnp.zeros, [embedding_dim]
    )
    new_observation = jnp.mean(flat_inputs, axis=0)
    if train:
      self._ema_update(feature_means, new_observation)
    return feature_means.value

  def update_stdev_estimate(self, flat_inputs, train):
    """Update an EMA estimate of the feature standard deviation."""
    feature_stdev = self.variable(
        'quantizer', 'feature_stdev', jnp.std, flat_inputs
    )
    new_observation = jnp.std(flat_inputs)
    if train:
      self._ema_update(feature_stdev, new_observation)
    return feature_stdev.value

  def _ema_update(self, variable, new_value):
    """Apply an EMA variable update, possibly in cross-device context."""
    if self.cross_replica_axis:
      new_value = jax.lax.psum(new_value, axis_name=self.cross_replica_axis)
    variable.value = (
        self.ema_decay * variable.value + (1.0 - self.ema_decay) * new_value
    )


class VectorQuantizer(BaseQuantizer):
  """Vector Quantizer using L2-loss.

  Attributes:
    commitment_loss: Loss weight for propagating quantization loss to inputs.
  """

  commitment_loss: float = 0.0
  demean: bool = False
  rescale: bool = False

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
    if self.demean:
      feature_means = self.update_mean_estimate(flat_inputs, train)
      flat_inputs -= feature_means
    if self.rescale:
      stdev = self.update_stdev_estimate(flat_inputs, train)
      flat_inputs /= stdev + 1e-8
    codebook = self.create_codebook(flat_inputs)

    # Find nearest neighbor indices.
    distances = (
        jnp.sum(jnp.square(flat_inputs), 1, keepdims=True)
        - 2 * jnp.matmul(flat_inputs, codebook.T)
        + jnp.sum(jnp.square(codebook.T), 0, keepdims=True)
    )
    nn_idx = jnp.argmin(distances, axis=1)
    encodings = jax.nn.one_hot(nn_idx, self.num_centroids)
    counts = self.update_cluster_counts(encodings, train)
    quantized = jnp.matmul(encodings, codebook)
    quantization_loss = self.loss(flat_inputs, quantized)
    quantization_loss = jnp.reshape(quantization_loss, inputs.shape)

    if self.rescale:
      quantized *= stdev + 1e-8
    if self.demean:
      quantized += feature_means
    quantized = jnp.reshape(quantized, inputs.shape)

    nn_idx = jnp.reshape(nn_idx, inputs.shape[:-1])

    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook_values = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook_values = jax.lax.stop_gradient(codebook_values)

    return QuantizerOutputs(
        quantized, quantization_loss, nn_idx, codebook_values, [counts]
    )


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
    quantization_loss = jnp.full(inputs.shape[:-1] + (1,), quantization_loss)
    # Apply stop gradient to protect the encodings from downstream losses.
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)

    # Expand the dimensions to match those of product quantizer, for interface
    # consistency. This can be seen as a product quantizer with just 1 section.
    nn_idx = jnp.expand_dims(nn_idx, 0)
    codebook_values = jnp.expand_dims(codebook, 0)

    if self.stop_gradient_codes:
      codebook_values = jax.lax.stop_gradient(codebook_values)

    return QuantizerOutputs(
        quantized, quantization_loss, nn_idx, codebook_values, [counts]
    )


class ProductQuantizer(nn.Module):
  """Product Quantizer.

  Attributes:
    num_sections: The number of sections to quantize.
    base_quantizers: A list of `num_sections` BaseQuantizer modules.
    stop_gradient_codes: Whether to apply a stop gradient on the quantizer's
      codes, to protect them from being modified by downstream losses. Should
      always be True, and a future CL with remove this from being an option
      (keeping only for comparisons purposes with previous code).
    pca_dim: Dimension for learned PCA projection. Set <= 0 to disable.
  """

  base_quantizers: Sequence[BaseQuantizer]
  stop_gradient_codes: bool = True
  pca_dim: int = 0

  def get_pca_layer(self, embedding_dim):
    """Create PCA params for projection and pre-bias."""
    if self.pca_dim <= 0:
      return jnp.ones([1]), jnp.zeros([1])
    projection = self.param(
        'pca_proj',
        jax.nn.initializers.variance_scaling(
            1.0, 'fan_avg', 'normal', dtype=jnp.float32
        ),
        [embedding_dim, self.pca_dim],
    )
    pre_bias = self.param(
        'pre_bias', jax.nn.initializers.zeros, [1, embedding_dim]
    )
    return projection, pre_bias

  def pca_project(self, flat_inputs):
    """Map to a low-dim'l space and minimize reconstruction error."""
    if self.pca_dim <= 0:
      return flat_inputs, 0, jnp.ones([1]), jnp.zeros([1])

    embedding_dim = flat_inputs.shape[-1]
    projection, pre_bias = self.get_pca_layer(embedding_dim)

    projected = jnp.matmul(flat_inputs + pre_bias, projection)
    unprojected = jnp.matmul(projected, projection.T) - pre_bias
    l2_loss = jnp.sqrt(jnp.sum(jnp.square(flat_inputs - unprojected), axis=-1))
    l2_loss = jnp.mean(l2_loss)

    # Ensure that (P@X)@(P@X).T is orthonormal.
    cov = jnp.matmul(projected.T, projected) / flat_inputs.shape[0]
    cov_loss = jnp.mean(jnp.square(cov - jnp.eye(self.pca_dim)))
    return projected, l2_loss + cov_loss, projection, pre_bias

  def pca_unproject(self, quantized, projection, pre_bias):
    if self.pca_dim <= 0:
      return quantized
    return jnp.matmul(quantized, projection.T) - pre_bias

  def get_num_centroids(self):
    nc = [q.num_centroids for q in self.base_quantizers]
    assert (
        len(list(set(nc))) == 1
    ), 'Expected all quantizers to have the same number of centroids.'
    return nc[0]

  def get_num_sections(self):
    return len(self.base_quantizers)

  @nn.compact
  def __call__(self, inputs, train):
    ns = self.get_num_sections()
    embedding_dim = inputs.shape[-1]
    flat_inputs = jnp.reshape(inputs, [-1, embedding_dim])
    flat_inputs, pca_loss, projection, pre_bias = self.pca_project(flat_inputs)

    # Divide the input into `num_sections` parts and quantize each separately.
    input_sections = jnp.split(flat_inputs, ns, axis=-1)
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
      loss.append(
          jnp.reshape(outputs.quantization_loss, inputs.shape[:-1] + (-1,))
      )
      counts += outputs.cluster_counts

    # Aggregate across 'sections' to get the following shapes:
    # quantized: [..., csz].
    # nn_idx: [ns, ...].
    # codebook: [ns, nc, csz / ns].
    quantized = jnp.concatenate(quantized, axis=-1)
    quantized = self.pca_unproject(quantized, projection, pre_bias)
    quantized = jnp.reshape(quantized, inputs.shape)
    nn_idx = jnp.concatenate(nn_idx, axis=0)
    nn_idx = jnp.reshape(nn_idx, (ns,) + inputs.shape[:-1])
    codebook = jnp.concatenate(codebook_list, axis=0)
    quantization_loss = jnp.mean(jnp.stack(loss, axis=0), axis=0) + pca_loss

    if self.stop_gradient_codes:
      codebook = jax.lax.stop_gradient(codebook)

    return QuantizerOutputs(
        quantized, quantization_loss, nn_idx, codebook, counts
    )


class ResidualQuantizer(nn.Module):
  """A residual quantizer with explicitly passed sub-quantizers.

  Accepting a list allows using arbitrary quantizers (e.g., product quantizers)
  in sequence.
  """

  quantizers: Sequence[nn.Module] = ()
  stop_gradient_codes: bool = True

  def get_num_centroids(self):
    nc = [q.num_centroids for q in self.quantizers]
    assert (
        len(list(set(nc))) == 1
    ), 'Expected all quantizers to have the same number of centroids.'
    return nc[0]

  def get_num_sections(self):
    return len(self.quantizers)

  @nn.compact
  def __call__(self, inputs, train=True):
    quantized = 0.0
    quantization_loss = 0.0
    nn_idx, codebooks, counts = [], [], []
    embedding_dim = inputs.shape[-1]

    flat_inputs = jnp.reshape(inputs, [-1, embedding_dim])
    residual = flat_inputs
    for quantizer in self.quantizers:
      quant_outputs = quantizer(residual, train)
      quantized += quant_outputs.quantized
      residual -= quant_outputs.quantized
      nn_idx.append(quant_outputs.nn_idx)
      codebooks.append(quant_outputs.codebook)
      quantization_loss += jnp.mean(quant_outputs.quantization_loss)
      counts += quant_outputs.cluster_counts

    # Aggregate across 'sections' to get the following shapes:
    # quantized: [...].
    # nn_idx: [ns, ...].
    # codebook: [ns, nc, csz / ns].
    # Using non-homogenous quantizers means we can't concat the outputs.
    nn_idx = jnp.concatenate(nn_idx, axis=0)
    nn_idx = jnp.reshape(nn_idx, (len(self.quantizers),) + inputs.shape[:-1])
    codebooks = jnp.concatenate(codebooks, axis=0)
    if self.stop_gradient_codes:
      codebooks = jax.lax.stop_gradient(codebooks)
    quantized = jnp.reshape(quantized, inputs.shape)
    quantization_loss = jnp.full(inputs.shape[:-1] + (1,), quantization_loss)
    return QuantizerOutputs(
        quantized, quantization_loss, nn_idx, codebooks, counts
    )
