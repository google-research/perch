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

"""Configuration to run baseline model on top of HuBERT embeddings."""
from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config


def get_config() -> config_dict.ConfigDict:
  """Create configuration dictionary for training."""
  sample_rate_hz = config_dict.FieldReference(32_000)
  batch_size = config_dict.FieldReference(64)
  target_class_list = config_dict.FieldReference("xenocanto")
  add_taxonomic_labels = config_dict.FieldReference(True)

  config = config_dict.ConfigDict()
  config.sample_rate_hz = sample_rate_hz
  config.batch_size = batch_size

  # Configure the data
  train_window_size = config_dict.FieldReference(5)
  eval_window_size = config_dict.FieldReference(5)
  frame_rate_hz = config_dict.FieldReference(100)
  num_channels = config_dict.FieldReference(160)

  config.train_window_size = train_window_size
  config.eval_window_size = eval_window_size
  config.frame_rate_hz = frame_rate_hz
  config.num_channels = num_channels

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
          _c("pipeline.RandomSlice", window_size=train_window_size),
          _c("pipeline.RandomNormalizeAudio", min_gain=0.15, max_gain=0.25),
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
          _c("pipeline.Batch", batch_size=batch_size,
             split_across_devices=True),
          _c("pipeline.Slice", window_size=eval_window_size, start=0.0),
          _c("pipeline.NormalizeAudio", target_gain=0.2),
      ])
  eval_dataset_config.split = "train"
  config.eval_dataset_config = eval_dataset_config

  # Configure the experiment setup
  init_config = config_dict.ConfigDict()
  init_config.learning_rate = 0.0001
  init_config.input_shape = ((train_window_size * sample_rate_hz).get(),)
  init_config.rng_seed = 0
  init_config.target_class_list = target_class_list
  config.init_config = init_config

  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      "efficientnet.EfficientNet",
      model=_c("efficientnet.EfficientNetModel", value="b1"))
  model_config.taxonomy_loss_weight = 0.001
  init_config.model_config = model_config

  # HuBERT's early feature extractor.
  conv_layer_tuples = tuple([(512, 10, 5), (512, 3, 2),
                             (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2),
                             (512, 2, 2)])
  early_fs = _c(
      "layers.EarlyFeatureExtractor",
      dropout_prob=0.,
      activation=config_utils.object_config("nn.gelu"),
      conv_layer_tuples=conv_layer_tuples,
      deprecated_group_conv=True)

  # HuBERT's early feature extractor.
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
  conformer_config.skip_layer_norm = True
  late_fs = _c("conformer.Conformer", conformer_config)

  # HuBERT's "feature extractor" (early feature extractor followed by
  # late feature extractor, up to the specified block).
  model_config.hubert_feature_extractor = _c(
      "hubert.HuBERTEval",
      early_feature_extractor=early_fs,
      late_feature_extractor=late_fs,
      frontend=None,
      use_raw_audio=True,
      add_positional_embeddings=False)
  model_config.frontend = None

  # Configure the training loop
  num_train_steps = config_dict.FieldReference(1_000_000)

  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = num_train_steps
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 25_000
  config.train_config = train_config

  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = num_train_steps
  eval_config.eval_steps_per_checkpoint = 1000
  eval_config.tflite_export = True
  eval_config.input_shape = ((eval_window_size * sample_rate_hz).get(),)
  config.eval_config = eval_config

  config.reload_hubert_from = ""

  return config
