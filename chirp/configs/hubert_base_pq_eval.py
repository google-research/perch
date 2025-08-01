# coding=utf-8
# Copyright 2025 The Perch Authors.
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
  config.init_config.workdir = ""
  return config


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep(
          "config.eval_config.train_mode_at_eval", hyper.discrete([True, False])
      ),
      hyper.sweep(
          "config.eval_config.mask_at_eval", hyper.discrete([True, False])
      ),
  ])
