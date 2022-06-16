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

"""Utilities to filter and scrub data."""
from typing import Any, Dict, List, Union
import numpy as np


def not_in(feature_dict: Dict[str, Any], key: str, values: List[Any]) -> bool:
  """Applies a filtering operation."""
  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field.')
  expected_type = type(feature_dict[key])
  if not isinstance(values[0], expected_type):
    raise TypeError('The type of values specified for filtering needs to match '
                    'the type of values in that Table.')
  if feature_dict[key] in values:
    return False
  return True


def scrub(feature_dict: Dict[str, Any],
          key: str,
          values: Union[List[Any], np.ndarray],
          all_but: bool = False) -> Dict[str, Any]:
  """Applies a scrubbing operation."""

  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field.')
  if type(feature_dict[key]) not in [list, np.ndarray]:
    raise ValueError('For now, only scrub from lists/ndarrays.')
  # values and feature_dict[key] could be ndarray -> not using the 'not values'
  if len(values) == 0 or len(feature_dict[key]) == 0:
    return feature_dict
  data_type = type(feature_dict[key][0])
  if not isinstance(values[0], data_type):
    raise TypeError(
        'Values have type {}, while array_values have type {}'.format(
            type(values[0]), data_type))
  # Avoid changing the feature_dict in-place.
  new_feature_dict = feature_dict.copy()
  if all_but:
    new_feature_dict[key] = [x for x in feature_dict[key] if x in values]
  else:
    new_feature_dict[key] = [x for x in feature_dict[key] if x not in values]
  return new_feature_dict
