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

"""Tests for EfficientNet."""
import operator
from chirp.models import efficientnet
from jax import numpy as jnp
from jax import random
from jax import tree_util


def test_efficientnet():
  efficientnet_ = efficientnet.EfficientNet(
      model=efficientnet.EfficientNetModel.B0, include_top=False)
  key = random.PRNGKey(0)
  params_key, dropout_key = random.split(key)
  inputs = jnp.ones((1, 224, 224, 3))
  out, variables = efficientnet_.init_with_output(
      {
          "dropout": dropout_key,
          "params": params_key
      }, inputs, train=True)
  assert out.shape == (1, 7, 7, 1280)
  num_parameters = tree_util.tree_reduce(
      operator.add, tree_util.tree_map(jnp.size, variables["params"]))
  # Keras has 7 more parameters due to the normalization of the inputs
  assert num_parameters == 4_007_548
