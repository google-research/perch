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

"""An extension of chirp's original taxonomy_model."""
from chirp.models import taxonomy_model


class TaxonomyModel(taxonomy_model.TaxonomyModel):
  """Adding parameters masking utility to chirp's original taxonomy_model."""

  @staticmethod
  def is_bn_parameter(parameter_name: list[str]) -> bool:
    """Verifies whether some parameter belong to a BatchNorm layer.

    Args:
      parameter_name: The name of the parameter, as a list in which each member
        describes the name of a layer. E.g. ('Block1', 'batch_norm_1', 'bias').

    Returns:
      True if this parameter belongs to a BatchNorm layer.
    """
    return any(['BatchNorm' in x for x in parameter_name])
