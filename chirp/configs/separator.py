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
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  config = config_dict.ConfigDict()
  config.batch_size = 128
  config.rng_seed = 0
  config.learning_rate = 0.0001
  config.sample_rate_hz = 32000

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = 2_000_000
  train_config.log_every_steps = 500
  train_config.checkpoint_every_steps = 10_000
  train_config.loss_max_snr = 30.0

  eval_config = config_dict.ConfigDict()
  eval_config.eval_steps_per_loop = 100
  eval_config.tflite_export = False

  data_config = config_dict.ConfigDict()
  data_config.window_size_s = 5
  data_config.min_gain = 0.15
  data_config.max_gain = 0.75

  eval_data_config = config_dict.ConfigDict()
  eval_data_config.window_size_s = 5
  eval_data_config.min_gain = 0.15
  eval_data_config.max_gain = 0.75

  stft_config = config_dict.ConfigDict()
  stft_config.sample_rate_hz = config.sample_rate_hz
  stft_config.frame_rate = 100
  stft_config.frame_length_secs = 0.08
  stft_config.use_tf_stft = True
  stft_config.mag_spec = False

  learned_fb_config = config_dict.ConfigDict()
  learned_fb_config.features = 128
  learned_fb_config.kernel_size = 128
  learned_fb_config.strides = 32

  soundstream_config = config_dict.ConfigDict()
  soundstream_config.base_filters = 128
  # Bottleneck filters has minimal impact on quality.
  soundstream_config.bottleneck_filters = 128
  soundstream_config.output_filters = 1024
  soundstream_config.num_residual_layers = 5
  soundstream_config.strides = (2, 5, 2)
  soundstream_config.feature_mults = (2, 2, 2)
  soundstream_config.groups = (1, 1, 1)

  model_config = config_dict.ConfigDict()
  model_config.num_mask_channels = 4
  model_config.mask_kernel_size = 3

  config.mask_generator_type = "soundstream_unet"
  config.bank_type = "learned"
  config.unbank_type = "learned"

  config.mask_generator_config = soundstream_config

  if config.bank_type == "learned":
    config.bank_transform_config = learned_fb_config
    config.unbank_transform_config = learned_fb_config
    model_config.bank_is_real = True
  elif model_config.bank_type == "stft":
    config.bank_transform_config = stft_config
    config.unbank_transform_config = stft_config
    model_config.bank_is_real = False

  config.train_config = train_config
  config.data_config = data_config
  config.eval_data_config = eval_data_config
  config.model_config = model_config
  config.eval_config = eval_config
  return config
