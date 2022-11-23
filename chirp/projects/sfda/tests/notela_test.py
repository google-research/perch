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

import functools
import itertools

from chirp.projects.sfda.methods import notela
import flax.linen as nn
import jax
import jax.numpy as jnp
from scipy import sparse

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
    compute_nearest_neighbor_fn = functools.partial(
        notela.NOTELA.compute_nearest_neighbors,
        batch_feature=batch_feature,
        dataset_feature=dataset_feature,
        knn=knn)

    def to_dense(matrix):
      if isinstance(matrix, jnp.ndarray):
        return matrix
      else:
        return matrix.todense()

    nearest_neighbors_matrices = []
    for (efficient, sparse_storage) in itertools.product((True, False),
                                                         (True, False)):
      nearest_neighbors_matrices.append(
          compute_nearest_neighbor_fn(
              sparse_storage=sparse_storage,
              memory_efficient_computation=efficient))

    nearest_neighbors_reference = to_dense(nearest_neighbors_matrices[0])
    self.assertEqual(nearest_neighbors_reference.shape,
                     (n_points_batch, n_points_dataset))
    self.assertTrue((nearest_neighbors_reference.sum(-1) >= 0).all())
    self.assertTrue((nearest_neighbors_reference.sum(-1) <= knn).all())

    for nn_matrix_version in nearest_neighbors_matrices[1:]:
      self.assertTrue(
          jnp.allclose(nearest_neighbors_reference,
                       to_dense(nn_matrix_version)))

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
    sparse_nn_matrix = sparse.csr_matrix(nn_matrix)
    pseudo_labels = notela.NOTELA.teacher_step(
        batch_proba=one_hot(batch_proba),
        dataset_proba=one_hot(dataset_proba),
        nn_matrix=nn_matrix,
        lambda_=lambda_,
        alpha=alpha,
        normalize_pseudo_labels=True,
    )
    pseudo_labels_from_sparse = notela.NOTELA.teacher_step(
        batch_proba=one_hot(batch_proba),
        dataset_proba=one_hot(dataset_proba),
        nn_matrix=sparse_nn_matrix,
        lambda_=lambda_,
        alpha=alpha,
        normalize_pseudo_labels=True,
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

  def test_pad_pseudo_label(self):
    key = jax.random.PRNGKey(0)
    num_samples = 3
    label_mask = jnp.array([0, 1, 0, 0, 1]).astype(bool)
    used_classes = label_mask.sum()
    pseudo_labels = jax.random.normal(key, (num_samples, used_classes))
    padded_pseudo_label = notela.NOTELA.pad_pseudo_label(
        label_mask, pseudo_labels)
    self.assertTrue((padded_pseudo_label[:, label_mask] == pseudo_labels).all())


if __name__ == "__main__":
  absltest.main()
