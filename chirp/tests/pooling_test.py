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

"""Tests for pooling."""
import functools

from chirp.models import pooling
from jax import numpy as jnp
from jax import random
import numpy as np

from absl.testing import absltest


class PoolingTest(absltest.TestCase):

  def test_gaussian_pooling(self):
    num_channels = 2
    num_steps = 100
    window_size = 50

    inputs = jnp.reshape(
        jnp.arange(num_channels * num_steps, dtype=jnp.float32),
        (num_channels, num_steps),
    )
    inputs = inputs.T[jnp.newaxis]

    window_pool = pooling.WindowPool(
        window=pooling.gaussian,
        window_size=window_size,
        window_init=functools.partial(pooling.gaussian_init, std=5.0),
        stride=window_size,
        padding="VALID",
    )

    rng = random.PRNGKey(0)
    variables = window_pool.init(rng, inputs)
    outputs = window_pool.apply(variables, inputs)

    self.assertEqual(outputs.shape, (1, num_steps // window_size, num_channels))

    np.testing.assert_allclose(outputs[0, 0, 0], (window_size - 1) / 2)


if __name__ == "__main__":
  absltest.main()
