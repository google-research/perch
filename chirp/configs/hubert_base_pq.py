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

"""Configuration to run HuBERT with Product Quantizers."""
from chirp import config_utils
from chirp.configs import hubert_presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""

  config = hubert_presets.get_base_config()

  # Configure the data
  config.train_dataset_config = hubert_presets.get_train_pipeline(
      config,
      mixin_prob=0.75,
      train_dataset_dir="bird_taxonomy/slice_peaked:1.4.0",
  )
  config.eval_dataset_config = hubert_presets.get_eval_pipeline(
      config, "soundscapes/caples:1.1.0"
  )
  # Configure the experiment setup
  config.init_config = hubert_presets.get_base_init_config(config)
  model_config = hubert_presets.get_model_config()
  config.init_config.model_config = model_config

  conformer_config = hubert_presets.get_conformer_config()
  model_config.late_feature_extractor = _c(
      "conformer.Conformer", conformer_config
  )

  early_fs_config = hubert_presets.get_early_fs_config()
  config.init_config.early_fs_config = early_fs_config

  mask_config = hubert_presets.get_mask_config()
  model_config.mask_config = mask_config

  classifier_config = hubert_presets.get_classifier_config()
  model_config.classifier_config = classifier_config

  quantizer_config = hubert_presets.get_quantizer_config()
  base_quantizer_config = hubert_presets.get_base_quantizer_config()
  config.init_config.quantizer_config = quantizer_config
  config.init_config.base_quantizer_config = base_quantizer_config

  frontend_config = hubert_presets.get_frontend_config(config)
  config.init_config.frontend_config = frontend_config

  config.train_config = hubert_presets.get_base_train_config(config)
  config.eval_config = hubert_presets.get_base_eval_config(config)

  config.export_config = config_dict.ConfigDict()
  config.export_config.input_shape = (
      config.get_ref("eval_window_size_s") * config.get_ref("sample_rate_hz"),
  )
  config.export_config.num_train_steps = config.get_ref("num_train_steps")

  return config


def get_hyper(hyper):
  return hyper.sweep(
      "config.init_config.learning_rate", hyper.discrete([0.0001])
  )
