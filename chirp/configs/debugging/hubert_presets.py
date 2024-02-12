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

"""HuBERT presets for the control experiments."""

from chirp.config_utils import callable_config as _c
from chirp.configs import hubert_presets as hubert_presets_default
from chirp.configs.debugging import presets as presets_debug
from ml_collections import config_dict


def get_base_config(**kwargs):
  """Create the base config object.

  Contains common values and FieldReferences.

  Args:
    **kwargs: Values to add or override in the base config.

  Returns:
    Config dict containing common default values.
  """
  config = presets_debug.get_base_config(**kwargs)
  config.batch_size = 128
  config.num_train_steps = 4_000_000
  config.num_quantizer_pretrain_steps = 0
  config.update(kwargs)
  return config


def get_base_init_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default init config."""
  init_config = presets_debug.get_base_init_config(config, **kwargs)
  init_config.learning_rate_schedule = 'piecewise_linear'
  init_config.learning_rate = 0.0001
  init_config.start_learning_rate = 0.000001
  init_config.quant_start_learning_rate = 1e-5
  init_config.reload_quantizer_from = ''
  init_config.reload_hubert_from = ''
  init_config.reload_hubert_omit_quantizers = False
  init_config.update(**kwargs)
  return init_config


def get_base_train_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default train config."""
  train_config = presets_debug.get_base_train_config(config, **kwargs)
  train_config.num_quantizer_pretrain_steps = config.get_ref(
      'num_quantizer_pretrain_steps'
  )
  train_config.readout_loss_mult = 100
  train_config.hubert_loss_mult = 1
  train_config.quant_loss_mult = 1
  train_config.update(**kwargs)
  return train_config


def get_base_eval_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  eval_config = presets_debug.get_base_eval_config(config, **kwargs)
  eval_config.train_mode_at_eval = False
  eval_config.mask_at_eval = False
  eval_config.update(**kwargs)
  return eval_config


def get_frontend_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Get the frontend config."""
  frontend_config = config_dict.ConfigDict()
  frontend_stride = config.get_ref('sample_rate_hz') // config.get_ref(
      'frame_rate_hz'
  )
  frontend_config.features = config.get_ref('num_channels')
  frontend_config.stride = frontend_stride
  # ~0.08 * 32,000 -- note: in previous HuBERT configs, this was 2_560
  frontend_config.kernel_size = 2_048
  frontend_config.sample_rate = config.get_ref('sample_rate_hz')
  frontend_config.freq_range = (60, 10_000)
  frontend_config.scaling_config = _c(
      'frontend.PCENScalingConfig',
      # Disable convolutional approximation
      conv_width=0,
      # Solution to 2*pi*tau/T = arccos(1 - s^2/(2 * (1 - s))) (prop III.1)
      # for tau = 1.5 ms and T = 60 ms
      smoothing_coef=0.145,
      gain=0.8,
      bias=10.0,
      root=4.0,
  )
  frontend_config.omit_frontend = False
  frontend_config.update(**kwargs)
  return frontend_config


def get_train_pipeline(
    config: config_dict.ConfigDict, train_dataset_dir: str
) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  return presets_debug.get_supervised_train_pipeline(config, train_dataset_dir)


def get_eval_pipeline(
    config: config_dict.ConfigDict, eval_dataset_dir: str | dict[str, str]
) -> config_dict.ConfigDict:
  """Create Caples eval data pipeline."""
  return presets_debug.get_supervised_eval_pipeline(config, eval_dataset_dir)


def get_conformer_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_conformer_config(**kwargs)


def get_early_fs_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_early_fs_config(**kwargs)


def get_mask_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_mask_config(**kwargs)


def get_classifier_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_classifier_config(**kwargs)


def get_quantizer_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_quantizer_config(**kwargs)


def get_base_quantizer_config(**kwargs) -> config_dict.ConfigDict:
  return hubert_presets_default.get_base_quantizer_config(**kwargs)


def get_model_config(**kwargs) -> config_dict.ConfigDict:
  model_config = hubert_presets_default.get_model_config(**kwargs)
  model_config.readout_points = [6]
  return model_config
