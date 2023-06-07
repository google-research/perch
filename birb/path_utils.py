# coding=utf-8
# Copyright 2023 The BIRB Authors.
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

"""Path utilities.

General utilities to help with handling paths.
"""
import os
from typing import Iterable
from absl import logging
from etils import epath


def get_absolute_epath(relative_path: str) -> epath.Path:
  """Returns the absolute epath.Path associated with the relative_path.

  Args:
    relative_path: The relative path (w.r.t. root) to the resource.

  Returns:
    The absolute path to the resource.
  """
  file_path = epath.Path(__file__).parent / relative_path
  return file_path


def listdir(relative_path: str) -> Iterable[str]:
  """List the contents of a directory in the BIRB project."""
  absolute_path = get_absolute_epath(relative_path).as_posix()
  return os.listdir(absolute_path)
