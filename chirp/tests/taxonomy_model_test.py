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

"""Tests for TaxonomyModel."""

from chirp.models import efficientnet
from chirp.models import frontend
from chirp.models import taxonomy_model
import flax
import jax
from jax import numpy as jnp
from jax import random
from absl.testing import absltest


class TaxonomyModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    sample_rate_hz = 11025
    self.input_size = sample_rate_hz
    self.model = taxonomy_model.TaxonomyModel(
        num_classes={"label": 10},
        encoder=efficientnet.EfficientNet(
            model=efficientnet.EfficientNetModel.B0
        ),
        frontend=frontend.MorletWaveletTransform(
            features=160,
            stride=sample_rate_hz // 100,
            kernel_size=2_048,
            sample_rate=sample_rate_hz,
            freq_range=(60, 10_000),
            scaling_config=frontend.PCENScalingConfig(),
        ),
        taxonomy_loss_weight=0.0,
    )
    self.key = random.PRNGKey(0)
    self.variables = self.model.init(
        self.key, jnp.zeros((1, self.input_size)), train=False
    )

  def test_dropout(self):
    """Ensure that two passes with train=True provide different outputs."""

    fake_audio = 10 * random.normal(self.key, (1, 11025))
    rng, key = random.split(self.key)

    output1 = self.model.apply(
        self.variables,
        fake_audio,
        train=True,
        use_running_average=True,
        rngs={"dropout": rng},
    )
    key, rng = random.split(key)
    output2 = self.model.apply(
        self.variables,
        fake_audio,
        train=True,
        use_running_average=True,
        rngs={"dropout": rng},
    )
    self.assertNotEqual(
        jnp.squeeze(output1["label"]).tolist(),
        jnp.squeeze(output2["label"]).tolist(),
    )

  def test_batch_norm(self):
    """Ensure that the state is updated by BN layers."""

    fake_audio = 10 * random.normal(self.key, (2, 11025))
    rng, _ = random.split(self.key)
    model_state, _ = flax.core.pop(self.variables, "params")
    _, updated_state = self.model.apply(
        self.variables,
        fake_audio,
        train=False,
        use_running_average=False,
        mutable=list(model_state.keys()),
        rngs={"dropout": rng},
    )
    for x, y in zip(
        jax.tree_util.tree_leaves(model_state["batch_stats"]),
        jax.tree_util.tree_leaves(updated_state["batch_stats"]),
    ):
      self.assertNotEqual(x.tolist(), y.tolist())


if __name__ == "__main__":
  absltest.main()
