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

"""Tests for losses used in SFDA.."""

from chirp.projects.sfda import losses
import flax.linen as nn
import jax
import jax.numpy as jnp

from absl.testing import absltest


class LossesTest(absltest.TestCase):

  def test_binary_entropy(self):
    num_classes = 5
    n_points = 10
    logits = jax.random.uniform(jax.random.PRNGKey(0), (n_points, num_classes))
    binary_probas = nn.sigmoid(logits)
    label_mask = jnp.array([1, 1, 0, 0, 1])

    # Test that masking works as intended.
    masked_ent = losses.label_binary_ent(
        probabilities=binary_probas,
        label_mask=jnp.tile(label_mask, (n_points, 1)),
        eps=0.)
    ent = losses.label_binary_ent(
        probabilities=binary_probas[:, label_mask.astype(bool)], eps=0.)
    self.assertAlmostEqual(masked_ent.mean(), ent.mean())

    # Test that binary entropies fall in the right range [0, log(2)]
    self.assertTrue((ent >= 0.).all())
    self.assertTrue((ent <= jnp.log(2)).all())

    # Test that entropy and cross-entropy are consitent
    ent = losses.label_binary_ent(
        probabilities=binary_probas, label=binary_probas)
    xent = losses.label_binary_xent(
        probabilities=binary_probas, label=binary_probas)
    self.assertAlmostEqual(ent.mean(), xent.mean(), delta=1e-7)

  def test_standard_entropy(self):
    num_classes = 5
    n_points = 10
    logits = jax.random.uniform(jax.random.PRNGKey(0), (n_points, num_classes))
    probabilities = nn.softmax(logits)
    ent = losses.label_ent(probabilities)

    # Test that multi-class entropies fall in the range [0, log(num_classes)]
    self.assertTrue((ent >= 0.).all())
    self.assertTrue((ent <= jnp.log(num_classes)).all())

    # Ensure that entropy and cross-entropy with self are same.
    self.assertAlmostEqual(
        ent.mean(),
        losses.label_xent(probabilities=probabilities,
                          label=probabilities).mean())


if __name__ == "__main__":
  absltest.main()
