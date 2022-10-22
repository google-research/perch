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

"""Configuration to run baseline model."""
from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)
  batch_size = config_dict.FieldReference(128)
  target_class_list = config_dict.FieldReference("xenocanto")
  add_taxonomic_labels = config_dict.FieldReference(True)

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz
  config.batch_size = batch_size

  # Configure the data
  train_window_size = config_dict.FieldReference(30)
  eval_window_size = config_dict.FieldReference(5)
  frame_rate_hz = config_dict.FieldReference(100)
  num_channels = config_dict.FieldReference(160)

  config.train_window_size = train_window_size
  config.eval_window_size = eval_window_size
  config.frame_rate_hz = frame_rate_hz
  config.num_channels = num_channels

  scaling_config = _c("frontend.LogScalingConfig")

  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.Shuffle", shuffle_buffer_size=128),
          _c("pipeline.OnlyJaxTypes"),
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace="ebird2021",
             target_class_list=target_class_list,
             add_taxonomic_labels=add_taxonomic_labels),
          _c("pipeline.Pad", pad_size=train_window_size),
          _c("pipeline.RandomSlice", window_size=train_window_size),
          _c("pipeline.MixAudio", mixin_prob=0.75),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.RandomNormalizeAudio", min_gain=0.15, max_gain=0.25),
          _c(
              "pipeline.MelSpectrogram",
              features=num_channels,
              stride=sample_rate_hz // frame_rate_hz,
              kernel_size=2_048,  # ~0.08 * 32,000
              sample_rate=sample_rate_hz,
              freq_range=(60, 10_000),
              scaling_config=scaling_config),
          _c("pipeline.Repeat")
      ])
  train_dataset_config.split = "train"
  config.train_dataset_config = train_dataset_config

  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      "pipeline.Pipeline",
      ops=[
          _c("pipeline.OnlyJaxTypes"),
          _c("pipeline.ConvertBirdTaxonomyLabels",
             source_namespace="ebird2021",
             target_class_list=target_class_list,
             add_taxonomic_labels=add_taxonomic_labels),
          _c("pipeline.Pad", pad_size=eval_window_size, random=False),
          _c("pipeline.Slice", window_size=eval_window_size, start=0.0),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
          _c(
              "pipeline.MelSpectrogram",
              features=num_channels,
              stride=sample_rate_hz // frame_rate_hz,
              kernel_size=2_048,  # ~0.08 * 32,000
              sample_rate=sample_rate_hz,
              freq_range=(60, 10_000),
              scaling_config=scaling_config)
      ])
  eval_dataset_config.split = "train"
  config.eval_dataset_config = eval_dataset_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.learning_rate = 0.0001
  init_config.input_shape = (train_window_size * frame_rate_hz, 160)
  init_config.rng_seed = 0
  init_config.target_class_list = target_class_list
  config.init_config = init_config

  model_config = config_dict.ConfigDict()
  # Aim to have output targets of 256, starting at 144
  s = (256 / 144)**(1 / 5)
  model_config.encoder = _c(
      "taxonomy_model.ConformerModel",
      downsample=[(2, s), (5, s), (8, s), (11, s), (14, s)],
      kernel_size=15)
  model_config.taxonomy_loss_weight = 0.0
  init_config.model_config = model_config

  # Configure the training loop
  num_train_steps = config_dict.FieldReference(1_000_000)

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 5_000
  config.train_config = train_config

  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = num_train_steps
  eval_config.eval_steps_per_checkpoint = 1000
  eval_config.tflite_export = True
  eval_config.input_shape = (eval_window_size * frame_rate_hz, 160)
  config.eval_config = eval_config

  return config
