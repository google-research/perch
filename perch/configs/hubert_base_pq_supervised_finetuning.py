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

"""Configuration to run HuBERT with Product Quantizers."""
from chirp import config_utils
from chirp.configs import hubert_base_pq
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = hubert_base_pq.get_config()

  config.init_config.learning_rate_schedule = "cosine_decay"
  config.init_config.start_learning_rate = 0.0001
  config.init_config.reload_hubert_from = ""

  # Decide on which layer to add the supervision.
  # note that index 6 is the last layer!
  config.init_config.model_config.omit_classifier_stop_grads = (6,)

  # Turn off the HuBERT and quantizer training.
  config.train_config.hubert_loss_mult = 0
  config.train_config.quant_loss_mult = 0

  return config


def get_hyper(hyper):
  return hyper.sweep(
      "config.init_config.start_learning_rate", hyper.discrete([0.0001])
  )
