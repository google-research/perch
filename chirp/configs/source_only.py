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

from chirp import config_utils
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:

  # Configure adaptation
  method_config = config_dict.ConfigDict()
  method_config.da_method = config_utils.callable_config(
      "source_only.SourceOnly")
  method_config.num_epochs = 0  # No adaptation is done anyways.
  method_config.learning_rate = 0.  # Not used. Only kept for compatibility.
  return method_config
