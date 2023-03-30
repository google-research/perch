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

"""Configuration to train the EfficientNet baseline."""
from chirp import config_utils
from chirp.configs.baselines import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_model_config() -> config_dict.ConfigDict:
  """Returns the model config."""
  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      'efficientnet.EfficientNet',
      model=_c('efficientnet.EfficientNetModel', value='b5'),
  )
  model_config.taxonomy_loss_weight = 1e-3
  model_config.frontend = None
  return model_config


def get_config() -> config_dict.ConfigDict:
  """Creates the configuration dictionary for training and evaluation."""
  config = presets.get_base_config()
  config.init_config = presets.get_base_init_config(config)
  config.init_config.model_config = get_model_config()

  config.train_config = presets.get_base_train_config(config)
  config.train_dataset_config = presets.get_base_train_dataset_config(config)
  config.eval_config = presets.get_base_eval_config(config)
  config.eval_dataset_config = {
      'powdermill': presets.get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,  # Unused because filtering_df_path=None.
          slice_method='strided_windows',
          slice_start=0.0,
          eval_dataset_dir='soundscapes/powdermill_full_length:1.3.0',
      ),
      'caples': presets.get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,  # Unused because filtering_df_path=None.
          slice_method='fixed',
          slice_start=0.0,
          eval_dataset_dir='soundscapes/caples:1.3.0',
      ),
  }

  return config


def get_hyper(hyper):
  """Defines the hyperparameter sweep."""
  return hyper.sweep(
      'config.init_config.learning_rate', hyper.discrete([1e-3, 1e-2])
  )
