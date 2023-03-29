# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Configuration to run baseline model."""
from chirp import config_utils
from chirp.projects.sfda import adapt
from chirp.projects.sfda import models
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""

  config = config_dict.ConfigDict()
  config.modality = adapt.Modality.IMAGE
  config.multi_label = False
  config.eval_every = 1  # in epochs

  config.batch_size_adaptation = 64
  config.batch_size_eval = 64

  init_config = config_dict.ConfigDict()
  init_config.rng_seed = 0
  init_config.target_class_list = "cifar10_corrupted"
  init_config.corruption_name = "gaussian_noise"
  init_config.corruption_severity = 5
  init_config.pretrained_model = True

  config.init_config = init_config

  model_config = config_dict.ConfigDict()
  model_config.encoder = models.ImageModelName.WIDERESNET
  config.model_config = model_config

  config.eval_mca_every = -1

  return config
