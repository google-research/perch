# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Tests for SFDA utilities at chirp/projects/sfda/method_utils.py."""

from chirp.projects.sfda import method_utils
import jax
import jax.numpy as jnp
from scipy.spatial import distance
from absl.testing import absltest


class MethodUtilsTest(absltest.TestCase):

  def test_cdist(self):
    """Ensure that our pairwise distance function produces expected results."""
    feature_dim = 3
    n_points_a = 2
    n_points_b = 3
    features_a = jax.random.normal(
        jax.random.PRNGKey(0), (n_points_a, feature_dim)
    )
    feature_b = jax.random.normal(
        jax.random.PRNGKey(1), (n_points_b, feature_dim)
    )

    pairwises_sqr_distances_ours = method_utils.jax_cdist(features_a, feature_b)
    pairwises_sqr_distances_scipy = distance.cdist(features_a, feature_b) ** 2
    self.assertTrue(
        jnp.allclose(
            pairwises_sqr_distances_scipy, pairwises_sqr_distances_ours
        )
    )


if __name__ == "__main__":
  absltest.main()
