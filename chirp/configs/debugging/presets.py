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

"""Presets for the control experiments."""

from chirp.config_utils import callable_config as _c
from chirp.config_utils import object_config as _o
from ml_collections import config_dict


def get_base_config(**kwargs):
  """Create the base config object.

  Contains common values and FieldReferences.

  Args:
    **kwargs: Values to add or override in the base config.

  Returns:
    Config dict containing common default values.
  """
  config = config_dict.ConfigDict()
  config.sample_rate_hz = 32_000
  config.train_window_size_s = 5
  config.eval_window_size_s = 5
  config.frame_rate_hz = 100
  config.num_channels = 160
  config.batch_size = 256
  config.target_class_list = 'xenocanto'
  config.num_train_steps = 200_000
  config.loss_fn = _o('optax.sigmoid_binary_cross_entropy')
  config.tfds_data_dir = ''
  config.update(kwargs)
  return config


def get_base_init_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default init config."""
  init_config = config_dict.ConfigDict()
  init_config.input_shape = (
      config.get_ref('train_window_size_s') * config.get_ref('sample_rate_hz'),
  )
  init_config.learning_rate = 0.0001
  init_config.rng_seed = 0
  init_config.target_class_list = config.get_ref('target_class_list')
  init_config.update(**kwargs)
  return init_config


def get_base_train_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = config.get_ref('num_train_steps')
  train_config.log_every_steps = 1_250
  train_config.checkpoint_every_steps = 5_000
  train_config.update(**kwargs)
  return train_config


def get_base_eval_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = config.get_ref('num_train_steps')
  eval_config.tflite_export = False
  eval_config.update(**kwargs)
  return eval_config


def get_pcen_melspec_config(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  frontend_stride = config.get_ref('sample_rate_hz') // config.get_ref(
      'frame_rate_hz'
  )
  return _c(
      'frontend.MelSpectrogram',
      features=config.get_ref('num_channels'),
      stride=frontend_stride,
      kernel_size=2_048,  # ~0.08 ms * 32,000 Hz
      sample_rate=config.get_ref('sample_rate_hz'),
      freq_range=(60, 10_000),
      # Settings from PCEN: Why and how
      scaling_config=_c(
          'frontend.PCENScalingConfig',
          # Disable convolutional approximation
          conv_width=0,
          # Solution to 2*pi*tau/T = arccos(1 - s^2/(2 * (1 - s))) (prop III.1)
          # for tau = 1.5 ms and T = 60 ms
          smoothing_coef=0.145,
          gain=0.8,
          bias=10.0,
          root=4.0,
      ),
  )


def get_supervised_train_pipeline(
    config: config_dict.ConfigDict,
    train_dataset_dir: str,
    mixup=False,
    random_slice=False,
    random_gain=False,
) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  if train_dataset_dir != 'bird_taxonomy/slice_peaked:1.4.0':
    raise ValueError('we assume training on XC')
  train_dataset_config = config_dict.ConfigDict()
  if random_slice:
    slice_op = _c(
        'pipeline.RandomSlice',
        window_size=config.get_ref('train_window_size_s'),
    )
  else:
    slice_op = _c(
        'pipeline.Slice',
        window_size=config.get_ref('train_window_size_s'),
        start=0.5,
    )
  ops = [
      _c('pipeline.Shuffle', shuffle_buffer_size=512),
      _c('pipeline.OnlyJaxTypes'),
      _c(
          'pipeline.ConvertBirdTaxonomyLabels',
          source_namespace='ebird2021',
          target_class_list=config.get_ref('target_class_list'),
          add_taxonomic_labels=False,
      ),
      _c(
          'pipeline.MixAudio',
          target_dist=(1.0, 0.5, 0.25, 0.25) if mixup else (1.0,),
      ),
      slice_op,
      _c(
          'pipeline.Batch',
          batch_size=config.get_ref('batch_size'),
          split_across_devices=True,
      ),
      _c('pipeline.Repeat'),
  ]
  if random_gain:
    ops.append(
        _c('pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25)
    )
  train_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=ops,
  )
  train_dataset_config.split = 'train'
  train_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  train_dataset_config.dataset_directory = train_dataset_dir
  return train_dataset_config


def get_supervised_eval_pipeline(
    config: config_dict.ConfigDict,
    eval_dataset_dir: str | dict[str, str],
    normalize=False,
) -> config_dict.ConfigDict:
  """Create Caples eval data pipeline."""
  if isinstance(eval_dataset_dir, dict):
    return config_dict.ConfigDict(
        {
            name: get_supervised_eval_pipeline(config, dataset_dir, normalize)
            for name, dataset_dir in eval_dataset_dir.items()
        }
    )
  eval_dataset_config = config_dict.ConfigDict()
  ops = [
      _c('pipeline.OnlyJaxTypes'),
      _c(
          'pipeline.ConvertBirdTaxonomyLabels',
          source_namespace='ebird2021',
          target_class_list=config.get_ref('target_class_list'),
          add_taxonomic_labels=False,
      ),
      _c(
          'pipeline.Slice',
          window_size=config.get_ref('eval_window_size_s'),
          start=0.5,
      ),
      _c(
          'pipeline.Batch',
          batch_size=config.get_ref('batch_size'),
          split_across_devices=True,
      ),
  ]
  if normalize:
    ops.append(_c('pipeline.NormalizeAudio', target_gain=0.2))
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=ops,
  )
  eval_dataset_config.split = 'train'
  eval_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  eval_dataset_config.dataset_directory = eval_dataset_dir
  return eval_dataset_config
