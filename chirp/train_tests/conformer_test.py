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

"""Tests for conformer."""
import operator

from chirp.models import conformer
from jax import numpy as jnp
from jax import random
from jax import tree_util

from absl.testing import absltest


class ConformerTest(absltest.TestCase):

  def test_conv_subsample(self):
    batch_size = 2
    time = 500
    freqs = 80
    features = 144
    channels = 1
    inputs = jnp.ones((batch_size, time, freqs, channels))
    subsample = conformer.ConvolutionalSubsampling(features=features)

    key = random.PRNGKey(0)
    outputs, variables = subsample.init_with_output(key, inputs, train=False)

    self.assertEqual(outputs.shape, (batch_size, time // 4, features))

    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables['params'])
    )
    expected_num_parameters = (
        3 * 3 * 144
        + 144
        + 3 * 3 * 144 * 144  # First conv layer
        + 144
        + freqs // 4 * 144 * 144  # Second conv layer
        + 144  # Projection layer
    )
    self.assertEqual(num_parameters, expected_num_parameters)


if __name__ == '__main__':
  absltest.main()
