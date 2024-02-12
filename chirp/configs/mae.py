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

"""Configuration to run baseline model."""
from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)
  batch_size = config_dict.FieldReference(768)

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz
  config.batch_size = batch_size

  # Configure the data
  window_size_s = config_dict.FieldReference(5)

  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.Shuffle", shuffle_buffer_size=512),
          _c("pipeline.OnlyKeep", names=["audio"]),
          _c(
              "pipeline.Batch", batch_size=batch_size, split_across_devices=True
          ),
          _c("pipeline.RandomSlice", window_size=window_size_s),
          _c(
              "pipeline.MelSpectrogram",
              features=160,
              stride=sample_rate_hz // 128,
              kernel_size=2_048,  # ~0.08 * 32,000
              sample_rate=sample_rate_hz,
              freq_range=(60, 10_000),
              scaling_config=_c("frontend.PCENScalingConfig", conv_width=256),
          ),
          _c("pipeline.AddChannel"),
          _c("pipeline.Repeat"),
      ],
  )
  train_dataset_config.split = "train"
  config.train_dataset_config = train_dataset_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.learning_rate = 0.0001
  init_config.input_shape = (640, 160, 1)
  init_config.rng_seed = 0
  config.init_config = init_config

  model_config = config_dict.ConfigDict()
  init_config.model_config = model_config

  # Configure the training loop
  num_train_steps = config_dict.FieldReference(1_000_000)

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 25_000
  config.train_config = train_config

  return config
