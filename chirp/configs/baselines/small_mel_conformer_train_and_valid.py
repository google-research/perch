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

"""Configuration to train the (small) Conformer baseline."""
from chirp import config_utils
from chirp.configs.baselines import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_model_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Returns the model config."""
  model_config = config_dict.ConfigDict()
  model_config.taxonomy_loss_weight = 1e-3
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
  config = presets.get_base_config(melspec_in_pipeline=False)
  config.init_config = presets.get_base_init_config(config)
  config.init_config.model_config = get_model_config(config)
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
  return hyper.product([
      hyper.sweep(
          'config.random_augmentations',
          hyper.discrete([False, True]),
      ),
      hyper.sweep(
          'config.init_config.model_config.taxonomy_loss_weight',
          hyper.discrete([0, 1e-3]),
      ),
      hyper.sweep(
          'config.cosine_alpha',
          # Without / with cosine decay for the learning rate.
          hyper.discrete([1.0, 0.0]),
      ),
      hyper.sweep(
          'config.init_config.learning_rate',
          # 10 ** np.linspace(-5, 1, 5)
          hyper.discrete([1e-05, 3.16e-4, 1e-2, 3.16e-1, 1e1]),
      ),
  ])
