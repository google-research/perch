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

"""Configuration to run baseline model.
from chirp import config_utils
from chirp.configs.debugging import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config(config_string: str | None = None) -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  if config_string not in (
      None,
      'random_slice',
      'random_gain',
      'mixup',
      'pcen',
      'conformer',
  ):
    raise ValueError('unexpected config string')
  config = presets.get_base_config()

  # Configure the data
  config.train_dataset_config = presets.get_supervised_train_pipeline(
      config,
      # TODO(bartvm): Recreate datasets with new labels?
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      mixup=(config_string == 'mixup'),
      random_slice=(config_string == 'random_slice'),
      random_gain=(config_string == 'random_gain'),
  )
  # TODO(bartvm): Add XC validation split
  config.eval_dataset_config = presets.get_supervised_eval_pipeline(
      config,
      {
          'caples': 'soundscapes/caples:1.1.0',
          'xc': 'bird_taxonomy/slice_peaked:1.4.0',
      },
      normalize=(config_string == 'random_gain'),
  )
  # Configure the experiment setup
  config.init_config = presets.get_base_init_config(config)
  config.init_config.optimizer = _c(
      'optax.adam', learning_rate=config.init_config.get_ref('learning_rate')
  )
  model_config = config_dict.ConfigDict()
  if config_string == 'conformer':
    s = (256 / 144) ** (1 / 5)
    model_config.encoder = _c(
        'taxonomy_model.ConformerModel',
        # Each downsample reduces time by a factor of 2.
        # An additional downsample by 4 happens in the ConvolutionalSubsampling.
        downsample=[(2, s), (5, s), (8, s), (11, s), (14, s)],
        features=256,
        kernel_size=32,
    )
  else:
    model_config.encoder = _c(
        'efficientnet.EfficientNet',
        model=_c('efficientnet.EfficientNetModel', value='b5'),
    )
  model_config.taxonomy_loss_weight = 0.0
  model_config.frontend = presets.get_pcen_melspec_config(config)
  config.init_config.model_config = model_config
  if config_string == 'pcen':
    frontend_stride = config.get_ref('sample_rate_hz') // config.get_ref(
        'frame_rate_hz'
    )
    config.init_config.model_config.frontend = _c(
        'frontend.MelSpectrogram',
        features=config.get_ref('num_channels'),
        stride=frontend_stride,
        kernel_size=2_048,  # ~0.08 ms * 32,000 Hz
        sample_rate=config.get_ref('sample_rate_hz'),
        freq_range=(60, 10_000),
        scaling_config=_c(
            'frontend.PCENScalingConfig',
            conv_width=0,
            smoothing_coef=0.1,
            gain=0.5,
            bias=2.0,
            root=2.0,
        ),
    )
  # Configure the training loop
  config.train_config = presets.get_base_train_config(config)
  config.eval_config = presets.get_base_eval_config(
      config,
      input_shape=(
          config.get_ref('eval_window_size_s')
          * config.get_ref('sample_rate_hz'),
      ),
  )
  return config


def get_hyper(hyper):
  return hyper.sweep(
      'config.init_config.learning_rate', hyper.discrete([1e-3, 1e-2])
  )
