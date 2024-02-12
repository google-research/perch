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

"""Configuration to train only a Product Quantizer."""
from chirp import config_utils
from chirp.configs import hubert_base_pq
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = hubert_base_pq.get_config()

  config.init_config.learning_rate_schedule = "cosine_decay"
  config.init_config.quant_start_learning_rate = 0.1
  config.init_config.model_config.readout_points = [0]
  config.train_config.readout_loss_mult = 0
  config.train_config.hubert_loss_mult = 0

  return config


def get_hyper(hyper):
  return hyper.sweep(
      "config.init_config.quant_start_learning_rate", hyper.discrete([0.1])
  )
