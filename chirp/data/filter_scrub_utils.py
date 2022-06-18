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
from typing import Any, Dict, List


def not_in(feature_dict: Dict[str, Any], key: str, values: List[Any]) -> bool:
  """Ensures if feature_dict[key] is not in values.

  Useful for filtering.

  Args:
    feature_dict: A dictionary that represents the row (=recording) to be
      potentially filtered in a DataFrame.
    key: The field from feature_dict used for filtering.
    values: The values that feature_dict[key] needs to be distinct of in order
      to trigger a True response.

  Returns:
    True if feature_dict[key] is not in values, False otherwise.
  """
  if key not in feature_dict:
    raise ValueError(f'{key} is not a correct field.')
  expected_type = type(feature_dict[key])
  for index, val in enumerate(values):
    if not isinstance(val, expected_type):
      raise TypeError(
          'Values[{}] has type {}, while feature_dict[{}] has type {}'.format(
              index, type(val), key, expected_type))
  return feature_dict[key] not in values
