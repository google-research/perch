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

"""Configuration to train the (small) Conformer baseline ablation."""
from chirp import config_utils
from chirp.configs.baselines import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_model_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Returns the model config."""
  model_config = config_dict.ConfigDict()
  model_config.taxonomy_loss_weight = 0.0
  model_config.frontend = presets.get_pcen_melspec_config(config)
  # Aim to have output targets of 256, starting at 144
  s = (256 / 144) ** (1 / 5)
  model_config.encoder = _c(
      'taxonomy_model.ConformerModel',
      # Each downsample reduces time by a factor of 2.
      # An additional downsample by 4 happens in the ConvolutionalSubsampling.
      downsample=[(2, s), (5, s), (8, s), (11, s), (14, s)],
      kernel_size=15,
      num_conformer_blocks=4,
  )
  return model_config


def get_config() -> config_dict.ConfigDict:
  """Creates the configuration dictionary for training and evaluation."""
  config = presets.get_base_config(
      melspec_in_pipeline=False, cosine_alpha=1.0, random_augmentations=True
  )
  config.init_config = presets.get_base_init_config(
      config, learning_rate=3.16e-4
  )
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
      hyper.sweep(
          'config.init_config.rng_seed', hyper.discrete([1238])
      ),
  ])
