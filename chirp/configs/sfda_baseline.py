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

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz

  # Configure the data
  batch_size = config_dict.FieldReference(64)
  window_size_s = config_dict.FieldReference(5)

  train_data_config = config_dict.ConfigDict()
  train_data_config.split = "train"
  train_data_config.shuffle = True
  config.train_data_config = train_data_config

  adaptation_data_config = config_dict.ConfigDict()
  adaptation_data_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.MultiHot"),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.RandomNormalizeAudio", min_gain=0.15, max_gain=0.25),
      ])
  adaptation_data_config.split = "train[:50%]"
  adaptation_data_config.shuffle = True

  config.adaptation_data_config = adaptation_data_config

  eval_data_config = config_dict.ConfigDict()
  eval_data_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.MultiHot"),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.Slice", window_size=window_size_s, start=0.5),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
      ])
  eval_data_config.split = "train[50%:]"
  eval_data_config.shuffle = False

  config.eval_data_config = eval_data_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.input_size = window_size_s * sample_rate_hz
  init_config.rng_seed = 0
  init_config.pretrained_ckpt_dir = ""

  # Configure model
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      "efficientnet.EfficientNet",
      model=_c("efficientnet.EfficientNetModel", value="b1"))
  model_config.taxonomy_loss_weight = 0.25
  model_config.frontend = _c(
      "frontend.MorletWaveletTransform",
      features=160,
      stride=sample_rate_hz // 100,
      kernel_size=2_048,  # ~0.025 * 32,000
      sample_rate=sample_rate_hz,
      freq_range=(60, 10_000),
      scaling_config=_c("frontend.PCENScalingConfig"))
  init_config.model_config = model_config

  config.init_config = init_config
  return config
