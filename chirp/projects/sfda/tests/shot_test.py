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

"""Tests for SHOT method."""

from chirp.projects.sfda.methods import shot
import flax.linen as nn
import jax
import numpy as np
from scipy.spatial import distance

from absl.testing import absltest


class ShotTest(absltest.TestCase):

  def original_pl(
      self, embeddings: np.ndarray, probabilities: np.ndarray, threshold=0.0
  ) -> np.ndarray:
    """The orignal implementation of SHOT's pseudo-labelling function.

    Taken from https://github.com/tim-learn/SHOT/blob/
    07d0c713e4882e83fded1aff2a447dff77856d64/object/image_target.py#L242.

    Args:
      embeddings: The model's embeddings.
      probabilities: The model's probabilities.
      threshold: A threshold to only keep classes with a certain number of
        samples (set to 0 in the original code).

    Returns:
      The hard pseudo-labels.
    """

    predict = np.argmax(probabilities, axis=-1)
    num_classes = probabilities.shape[1]
    aff = probabilities

    for _ in range(2):
      initc = aff.transpose().dot(embeddings)
      initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
      cls_count = np.eye(num_classes)[predict].sum(axis=0)
      labelset = np.where(cls_count >= threshold)[0]

      dd = distance.cdist(embeddings, initc[labelset])
      pred_label = dd.argmin(axis=1)
      predict = labelset[pred_label]

      aff = np.eye(num_classes)[predict]

    return predict.astype('int')

  def test_pseudo_label(self):
    """Ensure that our reimplementation of SHOT's pseudo-labelling is correct."""
    n_points, feature_dim, num_classes = 10, 100, 10
    fake_embeddings = jax.random.normal(
        jax.random.PRNGKey(57), (n_points, feature_dim)
    )
    fake_probabilities = nn.softmax(
        jax.random.normal(jax.random.PRNGKey(58), (n_points, num_classes))
    )

    pl_original = self.original_pl(
        np.array(fake_embeddings), np.array(fake_probabilities)
    )
    pl_ours = shot.SHOT.compute_pseudo_label(
        dataset_feature=fake_embeddings,
        dataset_probability=fake_probabilities,
        multi_label=False,
    )

    self.assertTrue(np.allclose(pl_original, pl_ours.argmax(-1)))


if __name__ == '__main__':
  absltest.main()
