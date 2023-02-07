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

"""Config file for Test-time entropy minimization.

Wang, Dequan, et al. "Tent: Fully test-time adaptation by entropy minimization."
arXiv preprint arXiv:2006.10726 (2020).
"""
from chirp import config_utils
from chirp.projects.sfda import model_utils
from ml_collections import config_dict


def get_image_config() -> config_dict.ConfigDict:  # pylint: disable=missing-function-docstring
  # Configure adaptation
  image_config = config_dict.ConfigDict()

  optimizer_cfg = config_dict.ConfigDict()
  optimizer_cfg.optimizer = "adam"
  optimizer_cfg.opt_kwargs = {"momentum": 0.9, "nesterov": True}
  optimizer_cfg.weight_decay = 0.0
  optimizer_cfg.learning_rate = 1e-4
  optimizer_cfg.learning_rate_decay = model_utils.LearningRateDecay.COSINE
  optimizer_cfg.mult_learning_rate_resnet_base = 1.0
  optimizer_cfg.trainable_params_strategy = model_utils.TrainableParams.BN
  image_config.optimizer_config = optimizer_cfg

  # Forward options
  image_config.num_epochs = 10
  image_config.use_dropout = False
  image_config.update_bn_statistics = True
  return image_config


def get_audio_config() -> config_dict.ConfigDict:  # pylint: disable=missing-function-docstring
  # Configure adaptation
  audio_config = config_dict.ConfigDict()

  optimizer_cfg = config_dict.ConfigDict()
  optimizer_cfg.optimizer = "adam"
  optimizer_cfg.opt_kwargs = {"momentum": 0.9, "nesterov": True}
  optimizer_cfg.weight_decay = 0.0
  optimizer_cfg.learning_rate = 1e-5
  optimizer_cfg.learning_rate_decay = model_utils.LearningRateDecay.COSINE
  optimizer_cfg.mult_learning_rate_resnet_base = 1.0
  optimizer_cfg.trainable_params_strategy = model_utils.TrainableParams.BN
  audio_config.optimizer_config = optimizer_cfg

  # Forward options
  audio_config.num_epochs = 10
  audio_config.use_dropout = True
  audio_config.update_bn_statistics = False
  return audio_config


def get_config() -> config_dict.ConfigDict:
  method_config = config_dict.ConfigDict()
  method_config.sfda_method = config_utils.callable_config("tent.Tent")
  method_config.audio = get_audio_config()
  method_config.image = get_image_config()
  return method_config
