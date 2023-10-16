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

"""Configuration for contrastive learning."""
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Get the configuration."""
  config = config_dict.ConfigDict()
  config.dataset = ""
  config.validation_dataset = ""
  config.source_class_list = "xenocanto_v3"
  config.window_size = 5
  config.hop_size = 2.5
  config.batch_size = 512
  config.worker_count = 8
  config.num_one_shot_samples = 600
  config.log_interval = 100
  config.validation_interval = 1000
  config.checkpoint_interval = 1000
  config.seed = 0
  config.sample_rate = 32_000
  config.learning_rate = 1e-3
  return config
