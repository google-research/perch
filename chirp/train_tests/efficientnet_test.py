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

"""Tests for EfficientNet."""
import operator

from chirp.models import efficientnet
from chirp.models import efficientnet_v2
from jax import numpy as jnp
from jax import random
from jax import tree_util

from absl.testing import absltest
from absl.testing import parameterized


class EfficientNetTest(parameterized.TestCase):

  @parameterized.parameters(("default",), ("qat",))
  def test_efficientnet(self, use_qat):
    efficientnet_ = efficientnet.EfficientNet(
        model=efficientnet.EfficientNetModel.B0,
        include_top=False,
        op_set=use_qat,
    )
    key = random.PRNGKey(0)
    params_key, dropout_key = random.split(key)
    inputs = jnp.ones((1, 224, 224, 3))
    out, variables = efficientnet_.init_with_output(
        {"dropout": dropout_key, "params": params_key}, inputs, train=True
    )
    self.assertEqual(out.shape, (1, 7, 7, 1280))
    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    # Keras has 7 more parameters due to the normalization of the inputs
    self.assertEqual(num_parameters, 4_007_548)

  @parameterized.parameters(("default",), ("qat",))
  def test_efficientnet_v2(self, op_set):
    efficientnet_ = efficientnet_v2.EfficientNetV2(
        model_name="efficientnetv2-s", include_top=False, op_set=op_set
    )
    key = random.PRNGKey(0)
    params_key, dropout_key = random.split(key)
    inputs = jnp.ones((1, 224, 224, 3))
    out, variables = efficientnet_.init_with_output(
        {"dropout": dropout_key, "params": params_key}, inputs, train=True
    )
    self.assertEqual(out.shape, (1, 7, 7, 1280))
    num_parameters = tree_util.tree_reduce(
        operator.add, tree_util.tree_map(jnp.size, variables["params"])
    )
    # Keras EfficientNetV2S : 20_331_360
    # Keras has ~150k more parameters: Seems to be due to Keras using 4 params
    # per activation in the batchnorm op, where jax uses 2 (bias and scale).
    # There are 153872 BatchNorm params in the Jax implementation. This exactly
    # accounts for the difference in parameters.
    self.assertEqual(num_parameters, 20_177_488)


if __name__ == "__main__":
  absltest.main()
