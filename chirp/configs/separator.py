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

"""Configuration to run baseline separation model."""

from chirp import config_utils
from chirp.configs import presets
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = presets.get_base_config(batch_size=8, num_train_steps=5_000_000)

  config.train_dataset_config = presets.get_supervised_train_pipeline(
      config,
      mixin_prob=1.0,
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
  )
  config.train_dataset_config.split = 'train[:99%]'

  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.OnlyJaxTypes'),
          _c(
              'pipeline.ConvertBirdTaxonomyLabels',
              source_namespace='ebird2021',
              target_class_list=config.get_ref('target_class_list'),
              add_taxonomic_labels=True,
          ),
          _c('pipeline.MixAudio', mixin_prob=1.0),
          _c(
              'pipeline.Batch',
              batch_size=config.batch_size,
              split_across_devices=True,
          ),
          _c(
              'pipeline.Slice',
              window_size=config.get_ref('eval_window_size_s'),
              start=0.0,
          ),
          _c('pipeline.NormalizeAudio', target_gain=0.45),
      ],
  )
  eval_dataset_config.split = 'train[99%:]'
  eval_dataset_config.tfds_data_dir = config.tfds_data_dir
  eval_dataset_config.dataset_directory = 'bird_taxonomy/slice_peaked:1.4.0'
  config.eval_dataset_config = eval_dataset_config

  # Experiment configuration
  config.init_config = presets.get_base_init_config(config)

  # Model Configuration
  model_config = config_dict.ConfigDict()
  model_config.num_mask_channels = 4
  model_config.mask_kernel_size = 3
  model_config.classify_bottleneck = True
  model_config.classify_pool_width = 50
  model_config.classify_stride = 50
  model_config.classify_features = 512
  config.init_config.model_config = model_config

  # Mask generator model configuration
  soundstream_config = config_dict.ConfigDict()
  soundstream_config.base_filters = 128
  # Bottleneck filters has minimal impact on quality.
  soundstream_config.bottleneck_filters = 128
  soundstream_config.output_filters = 1024
  soundstream_config.num_residual_layers = 5
  soundstream_config.strides = (5, 2, 2)
  soundstream_config.feature_mults = (2, 2, 2)
  soundstream_config.groups = (1, 1, 1)
  soundstream_config.unet_scalar = 1.0
  model_config.mask_generator = config_utils.callable_config(
      'soundstream_unet.SoundstreamUNet', soundstream_config
  )

  # Frontend configuration
  stride = config_dict.FieldReference(32)

  frontend_config = config_dict.ConfigDict()
  frontend_config.features = 128
  frontend_config.stride = stride

  inverse_frontend_config = config_dict.ConfigDict()
  inverse_frontend_config.stride = stride

  kernel_size = config_dict.FieldReference(128)
  frontend_config.kernel_size = kernel_size
  inverse_frontend_config.kernel_size = kernel_size
  model_config.bank_transform = config_utils.callable_config(
      'frontend.LearnedFrontend', frontend_config
  )
  model_config.unbank_transform = config_utils.callable_config(
      'frontend.InverseLearnedFrontend', inverse_frontend_config
  )
  model_config.bank_is_real = True

  # Training loop configuration
  config.train_config = presets.get_base_train_config(config)
  config.train_config.loss_max_snr = 30.0
  config.train_config.classify_bottleneck_weight = 100.0
  config.train_config.taxonomy_labels_weight = 1.0

  config.eval_config = presets.get_base_eval_config(config)
  config.eval_config.eval_steps_per_checkpoint = 100
  config.eval_config.loss_max_snr = config.train_config.get_ref('loss_max_snr')
  config.eval_config.taxonomy_labels_weight = config.train_config.get_ref(
      'taxonomy_labels_weight'
  )

  # Note: frame_size should be divisible by the product of all downsampling
  # strides in the model architecture (eg, 32 * 5 * 2 * 2 * 50, for
  # frontend_config.stride=32, and soundstream_config.strides=[5, 2, 2]),
  # and classify_stride=50.
  config.export_config = config_dict.ConfigDict()
  config.export_config.frame_size = 32000
  config.export_config.num_train_steps = config.get_ref('num_train_steps')

  return config


def get_hyper(hyper):
  return hyper.sweep(
      'config.init_config.model_config.num_mask_channels',
      hyper.discrete([6]),
  )
