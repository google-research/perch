# coding=utf-8
# Copyright 2024 The Perch Authors.
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

"""Tests for quantizers."""

from chirp.models import quantizers
import flax
import jax
from jax import numpy as jnp
import numpy as np
from absl.testing import absltest


class QuantizersTest(absltest.TestCase):

  def test_vector_quantizer(self):
    num_centroids = 2
    embedding_dim = 8
    vq = quantizers.VectorQuantizer(
        num_centroids=num_centroids,
        commitment_loss=0.0,
        ema_decay=0.99,
        demean=True,
        rescale=True,
    )
    key = jax.random.PRNGKey(17)
    rngs = {}
    rngs['params'], key = jax.random.split(key)

    inputs = jnp.ones([2, 4, embedding_dim])
    params = vq.init(rngs, inputs, train=False, mutable=True)
    # Check that the cluster assignment counts are all 1's at init time.
    np.testing.assert_allclose(
        params['quantizer']['cluster_counts'], jnp.ones([num_centroids])
    )

    # Update a few times.
    for _ in range(5):
      _, params = vq.apply(params, inputs, train=True, mutable=True)
    # We have quantized the same zero vector over five batches, eight times in
    # each batch. Check that we have correct EMA estimates of the assignment
    # counts and feature mean.
    expected = jnp.array([1.0, 1.0])
    expected_means = jnp.zeros([embedding_dim])
    for _ in range(5):
      expected = 0.99 * expected + 0.01 * jnp.array([8.0, 0.0])
      expected_means = 0.99 * expected_means + 0.01 * jnp.ones([embedding_dim])

    np.testing.assert_allclose(params['quantizer']['cluster_counts'], expected)
    np.testing.assert_allclose(
        params['quantizer']['feature_means'], expected_means
    )
    np.testing.assert_allclose(
        params['quantizer']['feature_stdev'], 0.0, atol=1e-6
    )

  def test_refresh_codebooks(self):
    num_centroids = 2
    embedding_dim = 8
    vq = quantizers.VectorQuantizer(
        num_centroids=num_centroids,
        commitment_loss=0.0,
        ema_decay=0.99,
        demean=True,
    )
    key = jax.random.PRNGKey(17)
    rngs = {}
    rngs['params'], key = jax.random.split(key)

    inputs = jnp.ones([2, 4, embedding_dim])
    params = vq.init(rngs, inputs, train=False, mutable=True)
    model_state, model_params = flax.core.pop(params, 'params')

    # Refresh with threshold 0.0, which should leave the params unchanged.
    updated_params, updated_state = quantizers.refresh_codebooks(
        model_params, model_state, key, 0.0
    )
    flat_params = flax.traverse_util.flatten_dict(model_params)
    flat_updated_params = flax.traverse_util.flatten_dict(updated_params)
    flat_state = flax.traverse_util.flatten_dict(model_state)
    flat_updated_state = flax.traverse_util.flatten_dict(updated_state)
    for k in flat_params:
      np.testing.assert_allclose(flat_params[k], flat_updated_params[k])
    for k in flat_state:
      np.testing.assert_allclose(flat_state[k], flat_updated_state[k])

    # Update the VQs to change the usage counts.
    _, params = vq.apply(params, inputs, train=True, mutable=True)
    # Refresh the codebooks with threshold 2.0, which should cause codebooks
    # to update.
    model_state, model_params = flax.core.pop(params, 'params')
    updated_params, updated_state = quantizers.refresh_codebooks(
        model_params, model_state, key, 2.0
    )
    flat_params = flax.traverse_util.flatten_dict(model_params)
    flat_updated_params = flax.traverse_util.flatten_dict(updated_params)
    flat_state = flax.traverse_util.flatten_dict(model_state)
    flat_updated_state = flax.traverse_util.flatten_dict(updated_state)
    for k in flat_params:
      diff = np.sum(np.abs(flat_params[k] - flat_updated_params[k]))
      self.assertGreater(diff, 0.1)

  def test_product_quantizer(self):
    num_centroids = 2
    embedding_dim = 16
    num_sections = 4
    base_quantizers = [
        quantizers.VectorQuantizer(
            num_centroids=num_centroids,
            commitment_loss=0.0,
            ema_decay=0.99,
            demean=True,
        )
        for _ in range(num_sections)
    ]
    pvq = quantizers.ProductQuantizer(base_quantizers, pca_dim=8)
    key = jax.random.PRNGKey(17)
    rngs = {}
    rngs['params'], key = jax.random.split(key)

    inputs = jnp.ones([2, 4, embedding_dim])
    params = pvq.init(rngs, inputs, train=False, mutable=True)
    # Just check that it runs for now.
    quantizer_outputs, _ = pvq.apply(params, inputs, train=True, mutable=True)
    # hubert_train.py expects the quantization loss to be of dim 3, e.g.
    # [batch size, num frames, num clusters].
    self.assertLen(quantizer_outputs.quantization_loss.shape, 3)
    self.assertSequenceEqual(quantizer_outputs.quantized.shape, inputs.shape)
    self.assertSequenceEqual(
        quantizer_outputs.nn_idx.shape, [num_sections, 2, 4]
    )

  def test_residual_quantizer(self):
    num_centroids = 2
    embedding_dim = 8
    num_sections = 4
    base_quantizers = [
        quantizers.VectorQuantizer(
            num_centroids=num_centroids,
            commitment_loss=0.0,
            ema_decay=0.99,
            demean=True,
        )
        for _ in range(num_sections)
    ]
    rvq = quantizers.ResidualQuantizer(base_quantizers)
    key = jax.random.PRNGKey(17)
    rngs = {}
    rngs['params'], key = jax.random.split(key)

    inputs = jnp.ones([2, 4, embedding_dim])
    params = rvq.init(rngs, inputs, train=False, mutable=True)
    # Just check that it runs for now.
    quantizer_outputs, _ = rvq.apply(params, inputs, train=True, mutable=True)
    # hubert_train.py expects the quantization loss to be of dim 3, e.g.
    # [batch size, num frames, num clusters].
    self.assertLen(quantizer_outputs.quantization_loss.shape, 3)
    self.assertSequenceEqual(quantizer_outputs.quantized.shape, inputs.shape)
    self.assertSequenceEqual(
        quantizer_outputs.nn_idx.shape, [num_sections, 2, 4]
    )


if __name__ == '__main__':
  absltest.main()
