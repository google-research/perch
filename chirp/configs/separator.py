# coding=utf-8
# Copyright 2022 The Chirp Authors.
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
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)
  batch_size = config_dict.FieldReference(128)
  target_class_list = config_dict.FieldReference("xenocanto")

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz
  config.batch_size = batch_size

  # Configure the data
  window_size_s = config_dict.FieldReference(5)

  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.Shuffle", shuffle_buffer_size=512),
          _c("pipeline.OnlyJaxTypes"),
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace="ebird2021",
             target_class_list=target_class_list,
             add_taxonomic_labels=True),
          _c("pipeline.MixAudio", mixin_prob=1.0),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.RandomSlice", window_size=window_size_s),
          _c("pipeline.RandomNormalizeAudio", min_gain=0.15, max_gain=0.75),
          _c("pipeline.Repeat"),
      ])
  train_dataset_config.split = "train[:99%]"
  config.train_dataset_config = train_dataset_config

  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.OnlyJaxTypes"),
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace="ebird2021",
             target_class_list=target_class_list,
             add_taxonomic_labels=True),
          _c("pipeline.MixAudio", mixin_prob=1.0),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.Slice", window_size=window_size_s, start=0.0),
          _c("pipeline.NormalizeAudio", target_gain=0.45),
      ])
  eval_dataset_config.split = "train[99%:]"
  config.eval_dataset_config = eval_dataset_config

  # Experiment configuration
  init_config = config_dict.ConfigDict()
  init_config.rng_seed = 0
  init_config.learning_rate = 0.0001
  init_config.input_size = sample_rate_hz * window_size_s
  init_config.target_class_list = target_class_list
  config.init_config = init_config

  # Model Configuration
  model_config = config_dict.ConfigDict()
  model_config.num_mask_channels = 4
  model_config.mask_kernel_size = 3
  model_config.classify_bottleneck = True
  model_config.classify_pool_width = 50
  model_config.classify_stride = 50
  model_config.classify_features = 512
  init_config.model_config = model_config

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
      "soundstream_unet.SoundstreamUNet", soundstream_config)

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
      "frontend.LearnedFrontend", frontend_config)
  model_config.unbank_transform = config_utils.callable_config(
      "frontend.InverseLearnedFrontend", inverse_frontend_config)
  model_config.bank_is_real = True

  # Training loop configuration
  num_train_steps = config_dict.FieldReference(5_000_000)

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.log_every_steps = 500
  train_config.checkpoint_every_steps = 25_000
  train_config.loss_max_snr = 30.0
  train_config.classify_bottleneck_weight = 100.0
  train_config.taxonomy_labels_weight = 1.0
  config.train_config = train_config

  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = num_train_steps
  eval_config.eval_steps_per_checkpoint = 100
  # Note: frame_size should be divisible by the product of all downsampling
  # strides in the model architecture (eg, 32 * 5 * 2 * 2 * 50, for
  # frontend_config.stride=32, and soundstream_config.strides=[5, 2, 2]),
  # and classify_stride=50.
  eval_config.frame_size = 32000
  eval_config.tflite_export = True
  config.eval_config = eval_config

  return config
