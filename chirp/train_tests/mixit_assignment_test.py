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

"""Least-squares MixIT assignments partition the estimated sources."""

from absl.testing import absltest
from absl.testing import parameterized
from chirp.models import metrics
import jax
import jax.numpy as jnp
import numpy as np


def signals():
  return jnp.array([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])


class MixitAssignmentTest(parameterized.TestCase):

  @parameterized.product(batch=[1, 3], mixtures=[1, 2])
  def test_tied_sources_are_assigned_exactly_once(self, batch, mixtures):
    estimate = jnp.tile(signals()[None, ...], (batch, 1, 1))
    reference = jnp.tile(estimate.sum(axis=1, keepdims=True), (1, mixtures, 1))
    for fn in (
        metrics.least_squares_mixit,
        jax.jit(metrics.least_squares_mixit),
    ):
      mixed, assignment = fn(reference, estimate)
      np.testing.assert_array_equal(
          assignment.sum(axis=-2), np.ones((batch, 2))
      )
      np.testing.assert_array_equal(mixed.sum(axis=-2), estimate.sum(axis=-2))
      np.testing.assert_array_equal(assignment[:, 0, :], np.ones((batch, 2)))
      np.testing.assert_array_equal(
          assignment[:, 1:, :], np.zeros((batch, mixtures - 1, 2))
      )

  def test_tied_assignment_cannot_give_perfect_reconstruction_by_reusing_audio(
      self,
  ):
    estimate = jnp.stack([signals()[0], jnp.zeros(4)])[None, ...]
    reference = jnp.repeat(estimate[:, :1, :], 2, axis=1)
    mixed, assignment = metrics.least_squares_mixit(reference, estimate)
    self.assertGreater(float(jnp.square(mixed - reference).sum()), 0.0)
    np.testing.assert_array_equal(assignment.sum(axis=-2), [[1.0, 1.0]])

  def test_zero_and_unused_sources_still_have_one_assignment(self):
    estimate = jnp.zeros((2, 3, 5))
    reference = jnp.ones((2, 2, 5))
    mixed, assignment = metrics.least_squares_mixit(reference, estimate)
    np.testing.assert_array_equal(assignment.sum(axis=-2), np.ones((2, 3)))
    np.testing.assert_array_equal(mixed, np.zeros((2, 2, 5)))

  def test_unique_assignments_preserve_original_mixtures(self):
    estimate = signals()[None, ...]
    reference = estimate[:, ::-1, :]
    mixed, assignment = metrics.least_squares_mixit(reference, estimate)
    np.testing.assert_array_equal(assignment, [[[0.0, 1.0], [1.0, 0.0]]])
    np.testing.assert_array_equal(mixed, reference)

  def test_gradients_do_not_duplicate_source_contributions(self):
    estimate = jnp.stack([signals()[0], jnp.zeros(4)])[None, ...]
    reference = jnp.repeat(estimate[:, :1, :], 2, axis=1)
    objective = lambda x: jnp.square(
        metrics.least_squares_mixit(reference, x)[0]
    ).sum()
    expected = jnp.broadcast_to(
        2 * estimate.sum(axis=-2, keepdims=True), estimate.shape
    )
    for grad in (jax.grad(objective), jax.jit(jax.grad(objective))):
      np.testing.assert_allclose(grad(estimate), expected, rtol=0, atol=0)

  def test_batched_and_vmapped_assignments_match(self):
    estimate = jnp.tile(signals()[None, ...], (3, 1, 1))
    reference = jnp.tile(estimate.sum(axis=1, keepdims=True), (1, 2, 1))
    actual = metrics.least_squares_mixit(reference, estimate)
    expected = jax.vmap(
        lambda r, e: metrics.least_squares_mixit(r[None], e[None])
    )(reference, estimate)
    for a, e in zip(actual, expected):
      np.testing.assert_array_equal(a, e[:, 0])


if __name__ == "__main__":
  absltest.main()
