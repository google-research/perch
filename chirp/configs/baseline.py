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

"""Configuration to run baseline model."""
from chirp import config_utils
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz

  # Configure the data
  batch_size = config_dict.FieldReference(64)
  window_size_s = config_dict.FieldReference(5)
  min_gain = config_dict.FieldReference(0.15)
  max_gain = config_dict.FieldReference(0.25)

  train_data_config = config_dict.ConfigDict()
  train_data_config.batch_size = batch_size
  train_data_config.window_size_s = window_size_s
  train_data_config.min_gain = min_gain
  train_data_config.max_gain = max_gain
  train_data_config.mixin_prob = 0.75
  config.train_data_config = train_data_config

  eval_data_config = config_dict.ConfigDict()
  eval_data_config.batch_size = batch_size
  eval_data_config.window_size_s = window_size_s
  eval_data_config.min_gain = (min_gain + max_gain) / 2
  eval_data_config.max_gain = (min_gain + max_gain) / 2
  eval_data_config.mixin_prob = 0.0
  config.eval_data_config = eval_data_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.learning_rate = 0.01
  init_config.input_size = window_size_s * sample_rate_hz
  init_config.rng_seed = 0
  config.init_config = init_config

  model_config = config_dict.ConfigDict()
  model_config.encoder = config_utils.callable_config(
      "efficientnet.EfficientNet",
      model=config_utils.callable_config(
          "efficientnet.EfficientNetModel", value="b1"))
  model_config.taxonomy_loss_weight = 0.25
  init_config.model_config = model_config

  model_config.frontend = config_utils.callable_config(
      "frontend.MelSpectrogram",
      features=160,
      stride=sample_rate_hz // 100,
      kernel_size=2_560,  # 0.08 * 32,000
      sample_rate=sample_rate_hz,
      freq_range=(60, 10_000),
      scaling_config=config_utils.callable_config("frontend.PCENScalingConfig"))

  # Configure the training loop
  num_train_steps = config_dict.FieldReference(250_000)

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 5_000
  config.train_config = train_config

  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = num_train_steps
  config.eval_config = eval_config

  return config
