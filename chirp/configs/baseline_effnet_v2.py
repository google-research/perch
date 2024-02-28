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
  config = presets.get_base_config()
  # Configure the data
  config.train_dataset_config = presets.get_supervised_train_pipeline(
      config,
      mixin_prob=0.75,
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
  )
  config.eval_dataset_config = presets.get_supervised_eval_pipeline(
      config, 'soundscapes/powdermill:1.3.0'
  )
  # Configure the experiment setup
  config.init_config = presets.get_classifier_init_config(config)
  config.init_config.optimizer = _c(
      'optax.adam', learning_rate=config.init_config.get_ref('learning_rate')
  )
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      'efficientnet_v2.EfficientNetV2',
      model_name='efficientnetv2-s',
      op_set='qat',
  )
  model_config.taxonomy_loss_weight = 0.001
  model_config.frontend = presets.get_new_pcen_melspec_config(config)
  config.init_config.model_config = model_config
  # Configure the training loop
  config.train_config = presets.get_base_train_config(config)
  config.eval_config = presets.get_base_eval_config(config)

  config.export_config = config_dict.ConfigDict()
  config.export_config.input_shape = (
      config.get_ref('eval_window_size_s') * config.get_ref('sample_rate_hz'),
  )
  config.export_config.num_train_steps = config.get_ref('num_train_steps')

  return config


def get_hyper(hyper):
  return hyper.sweep(
      'config.init_config.rng_seed',
      hyper.discrete([17, 42, 666]),
  )
