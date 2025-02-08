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

"""Tests for Bernoulli RBF."""

from chirp.projects import bernoulli_rbf
from flax import nnx
from jax import numpy as jnp
import numpy as np
import optax

from absl.testing import absltest


class BernoulliRbfTest(absltest.TestCase):

  def test_kernel_numerics(self):

    xs = jnp.array([[0, 1], [2, 3]])
    ys = jnp.array([[1, 0], [1, 1], [2, 3]])

    with self.subTest('unit_scale'):
      scales = jnp.array([1.0, 1.0])
      got = bernoulli_rbf.scaled_rbf_kernel(xs, ys, scales, 0.0)
      expect = jnp.array([
          [2.0, 1.0, 8.0],
          [10.0, 5.0, 0.0],
      ])
      np.testing.assert_array_equal(got.shape, (2, 3))
      np.testing.assert_array_equal(got, expect)

    with self.subTest('scaled'):
      scales = jnp.array([2.0, 3.0])
      got = bernoulli_rbf.scaled_rbf_kernel(xs, ys, scales, 0.0)
      expect = jnp.array([
          [13.0, 4.0, 52.0],
          [85.0, 40.0, 0.0],
      ])
      np.testing.assert_array_equal(got.shape, (2, 3))
      np.testing.assert_array_equal(got, expect)

  def test_split_labeled_data(self):
    data = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    labels = jnp.array([0, 1, 0, 1])
    pos, neg = bernoulli_rbf.BernoulliRBF.split_labeled_data(data, labels)
    np.testing.assert_array_equal(pos, jnp.array([[2, 3], [6, 7]]))
    np.testing.assert_array_equal(neg, jnp.array([[0, 1], [4, 5]]))

  def test_log_prob(self):
    data = jnp.array([[1, 0], [1, 1], [2, 3]])
    labels = jnp.array([0, 1, 0])
    model = bernoulli_rbf.BernoulliRBF(
        data,
        labels,
        rngs=nnx.Rngs(666),
        learn_feature_weights=False,
    )
    # Set unit scales and bias for simplicity.
    model.scales_pos = jnp.array([1.0, 1.0])
    model.scales_neg = jnp.array([1.0, 1.0])
    model.weight_bias = jnp.array([0.0, 0.0])

    got_log_prob, got_log_wt = model(jnp.array([[0, 1]]))
    # Squared distances to the data points are [2.0, 1.0, 8.0], and only
    # the second example is positive.
    # Then the positive example weight is [exp(-1)] and negative example weights
    # are [exp(-2), exp(-8)].
    # Then our predicted probability is:
    # exp(-1) / (exp(-1) + exp(-2) + exp(-8)) ~= 0.7306.
    # The log-of-sum-of-exponentials is log(exp(-1) + exp(-2) + exp(-8)) = -1.
    # The total weight is exp(-1) + exp(-2) + exp(-8) = 1.
    expect_log_prob = -1.0 + -np.log((np.exp(-1) + np.exp(-2) + np.exp(-8)))
    np.testing.assert_allclose(got_log_prob, expect_log_prob, atol=1e-5)
    expect_log_wt = np.log(np.exp(-1) + np.exp(-2) + np.exp(-8))
    np.testing.assert_allclose(got_log_wt, expect_log_wt, atol=1e-5)

  def test_train_step(self):
    data = jnp.array([[1, 0], [1, 1], [2, 3]])
    labels = jnp.array([0, 1, 0])
    model = bernoulli_rbf.BernoulliRBF(
        data,
        labels,
        rngs=nnx.Rngs(666),
        data_mean=None,
        data_std=None,
        learn_feature_weights=False,
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing
    loss = bernoulli_rbf.train_step(model, optimizer, mu=1.0)
    self.assertLess(loss, 2.0)


if __name__ == '__main__':
  absltest.main()
