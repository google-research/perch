# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""A toy model, used for testing/debugging purposes only."""
from chirp.models import output
from chirp.projects.sfda.models import image_model
from flax import linen as nn
import jax.numpy as jnp


class ConstantEncoderModel(image_model.ImageModel):
  """A toy model, used for testing/debugging purposes only.

  The model contains a trainable head. The encoder part simply returns the
  raw images, after spatial average-pooling.
  """

  num_classes: int

  @nn.compact
  def __call__(self, x, train: bool, use_running_average: bool):
    x = jnp.mean(x, axis=(1, 2))
    model_outputs = {}
    model_outputs['embedding'] = x
    x = nn.Dense(self.num_classes, dtype=jnp.float32)(x)
    model_outputs['label'] = x
    return output.ClassifierOutput(**model_outputs)

  @staticmethod
  def is_bn_parameter(parameter_name: list[str]) -> bool:
    """Verifies whether some parameter belong to a BatchNorm layer.

    Args:
      parameter_name: The name of the parameter, as a list in which each member
        describes the name of a layer. E.g. ('Block1', 'batch_norm_1', 'bias').

    Returns:
      Whether this parameter belongs to a BatchNorm layer.
    """
    return False
