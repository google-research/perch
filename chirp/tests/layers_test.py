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

"""Tests for layers."""
import operator

from chirp.models import layers
from jax import numpy as jnp
from jax import random
from jax import tree_util

from absl.testing import absltest
from absl.testing import parameterized


class LayersTest(parameterized.TestCase):

  def test_mbconv(self):
    # See table 2 in the MobileNetV2 paper
    mbconv = layers.MBConv(
        features=24,
        kernel_size=(3, 3),
        strides=2,
        expand_ratio=6,
    )
    key = random.PRNGKey(0)
    inputs = jnp.ones((1, 112, 112, 16))
    outputs, variables = mbconv.init_with_output(
        key, inputs, train=True, use_running_average=False
    )
    self.assertEqual(outputs.shape, (1, 56, 56, 24))

    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    expected_num_parameters = (
        16 * 6 * 16
        + 3 * 3 * 6 * 16  # Expansion
        + 16 * 6 * 24  # Depthwise separable convolution  # Reduction
    )
    self.assertEqual(num_parameters, expected_num_parameters)

  def test_fused_mbconv(self):
    # See table 2 in the MobileNetV2 paper
    fused_mbconv = layers.FusedMBConv(
        features=24,
        kernel_size=(3, 3),
        strides=(2, 2),
        expand_ratio=6,
    )
    key = random.PRNGKey(0)
    inputs = jnp.ones((1, 112, 112, 16))
    outputs, variables = fused_mbconv.init_with_output(
        key, inputs, train=True, use_running_average=False
    )
    self.assertEqual(outputs.shape, (1, 56, 56, 24))

    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    expected_num_parameters = (
        3 * 3 * 16 * 16 * 6 + 6 * 16 * 24  # Expansion  # Projection
    )
    self.assertEqual(num_parameters, expected_num_parameters)

  def test_squeeze_and_excitation(self):
    squeeze_and_excitation = layers.SqueezeAndExcitation()
    key = random.PRNGKey(0)
    inputs = jnp.ones((1, 112, 112, 16))
    outputs, variables = squeeze_and_excitation.init_with_output(key, inputs)
    self.assertEqual(outputs.shape, (1, 112, 112, 16))

    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    expected_num_parameters = (
        16 * 16 // 4 + 16 // 4 + 16 // 4 * 16 + 16  # Squeeze  # Excite
    )
    self.assertEqual(num_parameters, expected_num_parameters)


if __name__ == "__main__":
  absltest.main()
