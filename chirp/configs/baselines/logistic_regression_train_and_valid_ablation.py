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

"""Configuration to run the logistic regression baseline."""
from chirp import config_utils
from chirp.configs.baselines import presets
from ml_collections import config_dict

_c = config_utils.callable_config
_o = config_utils.object_config


def get_encoder_config() -> config_dict.ConfigDict:
  encoder_config = config_dict.ConfigDict()
  encoder_config.aggregation = 'avg_pool'
  encoder_config.compute_mfccs = False
  encoder_config.num_mfccs = 20  # Unused by default.
  return encoder_config


def get_model_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Returns the model config."""
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      'handcrafted_features.HandcraftedFeatures',
      compute_mfccs=config.encoder_config.get_ref('compute_mfccs'),
      num_mfccs=config.encoder_config.get_ref('num_mfccs'),
      aggregation=config.encoder_config.get_ref('aggregation'),
      window_size=10,
      window_stride=10,
  )
  model_config.taxonomy_loss_weight = 0.0
  model_config.frontend = presets.get_pcen_melspec_config(config)
  return model_config


def get_config() -> config_dict.ConfigDict:
  """Creates the configuration dictionary for training and evaluation."""
  config = presets.get_base_config(
      batch_size=64,
      melspec_in_pipeline=False,
      random_augmentations=True,
      cosine_alpha=0.0,
  )
  config.encoder_config = get_encoder_config()
  config.init_config = presets.get_base_init_config(config, learning_rate=0.316)
  config.init_config.model_config = get_model_config(config)

  config.train_config = presets.get_base_train_config(config)
  config.train_dataset_config = presets.get_ablation_train_dataset_config(
      config
  )
  config.eval_config = presets.get_base_eval_config(config)
  config.eval_dataset_config = presets.get_ablation_eval_dataset_config(config)

  return config


def get_hyper(hyper):
  """Defines the hyperparameter sweep."""
  return hyper.product([
      hyper.sweep('config.init_config.rng_seed', hyper.discrete([1236])),
  ])
