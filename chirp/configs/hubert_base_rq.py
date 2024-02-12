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

"""Configuration to run HuBERT with Product Quantizers."""
from chirp import config_utils
from chirp.configs import hubert_base_pq
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = hubert_base_pq.get_config()

  config.init_config.base_quantizer_config.num_centroids = 1024

  config.init_config.quantizer_config.num_sections = 2
  config.init_config.quantizer_config.strategy = "residual_quantization"
  config.init_config.quantizer_config.use_entropy_quantizer = False

  return config
