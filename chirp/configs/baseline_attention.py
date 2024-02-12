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
from chirp.configs import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = presets.get_base_config(
      train_window_size_s=30,
      eval_window_size_s=5,
      frame_rate_hz=100,
      num_channels=160,
      pad_mask=True,
  )

  config.train_dataset_config = presets.get_supervised_train_pipeline(
      config,
      mixin_prob=0.75,
      train_dataset_dir='bird_taxonomy/full_length:1.4.0',
  )
  config.eval_dataset_config = presets.get_supervised_eval_pipeline(
      config, 'soundscapes/caples:1.1.0'
  )

  # Configure the experiment setup
  config.init_config = presets.get_classifier_init_config(config)
  config.init_config.optimizer = _c(
      'optax.adam', learning_rate=config.init_config.get_ref('learning_rate')
  )

  model_config = config_dict.ConfigDict()
  dim = 512
  model_config.frontend = _c(
      'layers.EarlyFeatureExtractor',
      conv_layer_tuples=(
          (dim, 20, 10),
          (dim, 3, 2),
          (dim, 3, 2),
          (dim, 3, 2),
          (dim, 3, 2),
          (dim, 2, 2),
          (160, 2, 2),
      ),
  )
  # Aim to have output targets of 256, starting at 144
  s = (256 / 144) ** (1 / 5)
  model_config.encoder = _c(
      'taxonomy_model.ConformerModel',
      downsample=[(2, s), (5, s), (8, s), (11, s), (14, s)],
      kernel_size=15,
  )
  model_config.taxonomy_loss_weight = 0.0
  config.init_config.model_config = model_config

  # Configure the training loop
  config.train_config = presets.get_base_train_config(config)
  input_shape = (
      config.get_ref('eval_window_size_s') * config.get_ref('sample_rate_hz'),
  )
  config.eval_config = presets.get_base_eval_config(
      config, input_shape=input_shape
  )
  return config
