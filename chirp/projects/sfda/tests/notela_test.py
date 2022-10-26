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

"""Tests for NOTELA method."""

from chirp.projects.sfda.methods import notela
import flax.linen as nn
import jax
from jax.experimental import sparse
import jax.numpy as jnp

from absl.testing import absltest


class NOTELATest(absltest.TestCase):

  def test_nn_matrix(self):
    """Ensure that NOTELA's nearest_neighbors matrix contains valid values."""
    feature_dim = 5
    n_points_batch = 2
    n_points_dataset = 3
    knn = 2
    key = jax.random.PRNGKey(0)
    batch_feature = jax.random.normal(key, (n_points_batch, feature_dim))
    dataset_feature = jax.random.normal(key, (n_points_dataset, feature_dim))
    nearest_neighbors = notela.NOTELA.compute_nearest_neighbors(
        batch_feature=batch_feature,
        dataset_feature=dataset_feature,
        knn=knn,
        sparse_storage=False)
    sparse_nearest_neighbors = notela.NOTELA.compute_nearest_neighbors(
        batch_feature=batch_feature,
        dataset_feature=dataset_feature,
        knn=knn,
        sparse_storage=True)
    self.assertEqual(nearest_neighbors.shape,
                     (n_points_batch, n_points_dataset))
    self.assertTrue((nearest_neighbors.sum(-1) >= 0).all())
    self.assertTrue((nearest_neighbors.sum(-1) <= knn).all())
    self.assertTrue(
        jnp.allclose(sparse_nearest_neighbors.todense(), nearest_neighbors))

  def test_teacher_step(self):
    """Ensure that NOTELA's teacher-step produces valid pseudo-labels."""
    n_points_batch = 2
    lambda_ = 1.0
    alpha = 1.0
    n_points_dataset = 3
    key = jax.random.PRNGKey(0)

    def one_hot(probas):
      return jnp.stack([1 - probas, probas], axis=-1)

    batch_proba = nn.sigmoid(jax.random.normal(key, (n_points_batch,)))
    dataset_proba = nn.sigmoid(jax.random.normal(key, (n_points_dataset,)))
    nn_matrix = jax.random.randint(key, (n_points_batch, n_points_dataset), 0,
                                   2)
    sparse_nn_matrix = sparse.BCOO.fromdense(nn_matrix)
    pseudo_labels = notela.NOTELA.teacher_step(
        batch_proba=one_hot(batch_proba),
        dataset_proba=one_hot(dataset_proba),
        nn_matrix=nn_matrix,
        lambda_=lambda_,
        alpha=alpha,
    )
    pseudo_labels_from_sparse = notela.NOTELA.teacher_step(
        batch_proba=one_hot(batch_proba),
        dataset_proba=one_hot(dataset_proba),
        nn_matrix=sparse_nn_matrix,
        lambda_=lambda_,
        alpha=alpha,
    )
    self.assertTrue(
        jnp.allclose(
            pseudo_labels.sum(-1),
            jnp.ones_like(pseudo_labels.sum(-1)),
            atol=1e-4))
    self.assertTrue(jnp.allclose(
        pseudo_labels,
        pseudo_labels_from_sparse,
    ))


if __name__ == "__main__":
  absltest.main()
