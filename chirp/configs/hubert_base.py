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

"""Configuration to run HuBERT."""
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
             add_taxonomic_labels=add_taxonomic_labels),
          _c("pipeline.MixAudio", mixin_prob=0.75),
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.RandomSlice", window_size=window_size_s),
          _c("pipeline.RandomNormalizeAudio", min_gain=0.15, max_gain=0.25),
          _c("pipeline.Repeat"),
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
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.Slice", window_size=window_size_s, start=0.0),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
      ])
  eval_dataset_config.split = "train"
  config.eval_dataset_config = eval_dataset_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.learning_rate = 5e-4  # This is the peak. Should linearly increase
  init_config.start_learning_rate = 5e-8  # This is the start of the increase.
  init_config.quant_start_learning_rate = 1e-5
  init_config.input_size = window_size_s * sample_rate_hz
  init_config.rng_seed = 0
  init_config.target_class_list = target_class_list
  config.init_config = init_config

  # Configure the conformer which is used as HuBERT's default encoder.
  conformer_config = config_dict.ConfigDict()
  conformer_config.model_dims = 768
  conformer_config.kernel_size = 32
  conformer_config.ff_activation = config_utils.object_config("nn.swish")
  conformer_config.ff_residual_weight = 0.5
  conformer_config.ffn_dim_multiplier = 4
  conformer_config.atten_num_heads = 8
  conformer_config.layer_order = "mhsa_before_conv"
  conformer_config.dropout_prob = 0.
  conformer_config.conv_residual_dropout = None
  conformer_config.atten_residual_dropout = None
  conformer_config.ffn_residual_dropout = None
  conformer_config.atten_dropout = None
  conformer_config.ffn_relu_dropout = None
  conformer_config.fflayer_weight_sharing = False
  conformer_config.num_blocks = 12
  model_config = config_dict.ConfigDict()
  model_config.late_feature_extractor = config_utils.callable_config(
      "conformer.Conformer", conformer_config)

  # With this configuration, the number of frames is reduced from 500 to 125,
  # and the framerate is reduced from 100Hz to 25Hz.
  # TODO(etriantafillou): Experiment with moving the second strided layer to
  # later, and consider increasing its kernel size to 5 to give it more context.
  early_fs_config = config_dict.ConfigDict()
  early_fs_config.omit_earlyfs = False
  early_fs_config.dropout_prob = 0.
  early_fs_config.activation = config_utils.object_config("nn.gelu")
  early_fs_config.num_frames = 125
  init_config.early_fs_config = early_fs_config

  # Configure the masking parameters.
  mask_config = config_dict.ConfigDict()
  mask_config.mask_prob = 0.08
  mask_config.mask_length = 10
  mask_config.min_masks = 1
  model_config.mask_config = mask_config

  # Configure the classifier parameters.
  classifier_config = config_dict.ConfigDict()
  classifier_config.classify_from_all = True
  classifier_config.per_frame_predictions = True
  classifier_config.classify_pool_width = 50
  classifier_config.classify_stride = 50
  classifier_config.classify_features = 512
  classifier_config.reduction_type = "MIDPOINT"
  model_config.classifier_config = classifier_config

  # Configure the quantizer parameters.
  base_quantizer_config = config_dict.ConfigDict()
  base_quantizer_config.num_centroids = 128
  base_quantizer_config.gamma = 1e-5
  quantizer_config = config_dict.ConfigDict()
  quantizer_config.num_sections = 8
  init_config.quantizer_config = quantizer_config
  init_config.base_quantizer_config = base_quantizer_config
  init_config.reload_quantizer_from = ""

  # Configure the frontend parameters.
  model_config.frontend = config_utils.callable_config(
      "frontend.MelSpectrogram",
      features=160,
      stride=sample_rate_hz // 100,
      kernel_size=2_560,  # 0.08 * 32,000
      sample_rate=sample_rate_hz,
      freq_range=(60, 10_000),
      scaling_config=config_utils.callable_config("frontend.PCENScalingConfig"))

  # Configure HuBERT.
  model_config.final_dim = 256
  model_config.logit_temp = 0.1
  # model_config.alpha = 1.0
  model_config.alpha = 0.5  # gets loss for both masked and unmasked
  model_config.taxonomy_loss_weight = 0.25
  model_config.readout_points = [2, 4, 6, 8, 10]
  init_config.model_config = model_config

  # Configure the training loop.
  # HuBERT trains in two stages: for 250K steps and 400K steps, respectively.
  # Uses 32 GPUs, with a batch size of at most 87.5 seconds of audio per GPU.
  num_train_steps = config_dict.FieldReference(2_000_000)
  num_quantizer_pretrain_steps = config_dict.FieldReference(0)
  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.num_quantizer_pretrain_steps = num_quantizer_pretrain_steps
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 5_000
  train_config.readout_loss_mult = 1
  config.train_config = train_config

  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = num_train_steps
  config.eval_config = eval_config

  return config
