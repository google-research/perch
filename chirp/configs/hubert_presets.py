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

"""Preset Configurations.

Philosophy:
* The base_config contains values which will be re-used throughout the main
  configuration.
* Refer to values in the base_config using `get_ref` to ensure that the values
  update correctly when performing hyperparameter sweeps or testing.
* The config_utils.parse_config resolves all references, so that downstream
  code doesn't panic.
"""

from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config


def get_base_config(**kwargs):
  """Create the base config object.

  Contains common values and FieldReferences.

  Args:
    **kwargs: Values to add or override in the base config.

  Returns:
    Config dict containing common default values.
  """
  config = config_dict.ConfigDict()
  config.sample_rate_hz = 32000
  config.train_window_size_s = 5
  config.eval_window_size_s = 5
  config.frame_rate_hz = 100
  config.num_channels = 160
  config.batch_size = 128
  config.add_taxonomic_labels = True
  config.target_class_list = 'xenocanto'
  config.num_train_steps = 4_000_000
  config.num_quantizer_pretrain_steps = 0
  config.pad_mask = False
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
  init_config.learning_rate_schedule = 'piecewise_linear'
  init_config.learning_rate = 0.0001
  init_config.start_learning_rate = 0.000001
  init_config.quant_start_learning_rate = 1e-5
  init_config.rng_seed = 0
  init_config.target_class_list = config.get_ref('target_class_list')
  init_config.reload_quantizer_from = ''
  init_config.reload_hubert_from = ''
  init_config.reload_hubert_omit_quantizers = False
  init_config.update(**kwargs)
  return init_config


def get_base_train_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default train config."""
  train_config = config_dict.ConfigDict()
  train_config.num_train_steps = config.get_ref('num_train_steps')
  train_config.num_quantizer_pretrain_steps = config.get_ref(
      'num_quantizer_pretrain_steps'
  )
  train_config.log_every_steps = 250
  train_config.checkpoint_every_steps = 25_000
  train_config.readout_loss_mult = 100
  train_config.hubert_loss_mult = 1
  train_config.quant_loss_mult = 1
  train_config.update(**kwargs)
  return train_config


def get_base_eval_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default eval config."""
  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = config.get_ref('num_train_steps')
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
  frontend_config.scaling_config = config_utils.callable_config(
      'frontend.PCENScalingConfig',
      conv_width=256,
  )
  frontend_config.omit_frontend = False
  frontend_config.update(**kwargs)
  return frontend_config


def get_train_pipeline(
    config: config_dict.ConfigDict, mixin_prob: float, train_dataset_dir: str
) -> config_dict.ConfigDict:
  """Create the supervised training data pipeline."""
  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.Shuffle', shuffle_buffer_size=512),
          _c('pipeline.OnlyJaxTypes'),
          _c(
              'pipeline.ConvertBirdTaxonomyLabels',
              source_namespace='ebird2021',
              target_class_list=config.get_ref('target_class_list'),
              add_taxonomic_labels=config.get_ref('add_taxonomic_labels'),
          ),
          _c('pipeline.MixAudio', mixin_prob=mixin_prob),
          _c(
              'pipeline.Pad',
              pad_size=config.get_ref('train_window_size_s'),
              add_mask=config.get_ref('pad_mask'),
          ),
          _c(
              'pipeline.RandomSlice',
              window_size=config.get_ref('train_window_size_s'),
          ),
          _c(
              'pipeline.Batch',
              batch_size=config.get_ref('batch_size'),
              split_across_devices=True,
          ),
          _c('pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25),
          _c('pipeline.Repeat'),
      ],
  )
  train_dataset_config.split = 'train'
  train_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  train_dataset_config.dataset_directory = train_dataset_dir
  return train_dataset_config


def get_eval_pipeline(
    config: config_dict.ConfigDict, eval_dataset_dir: str
) -> config_dict.ConfigDict:
  """Create Caples eval data pipeline."""
  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=[
          _c('pipeline.OnlyJaxTypes'),
          _c(
              'pipeline.ConvertBirdTaxonomyLabels',
              source_namespace='ebird2021',
              target_class_list=config.get_ref('target_class_list'),
              add_taxonomic_labels=config.get_ref('add_taxonomic_labels'),
          ),
          _c(
              'pipeline.Pad',
              pad_size=config.get_ref('eval_window_size_s'),
              random=False,
              add_mask=config.get_ref('pad_mask'),
          ),
          _c(
              'pipeline.Slice',
              window_size=config.get_ref('eval_window_size_s'),
              start=0.0,
          ),
          _c(
              'pipeline.Batch',
              batch_size=config.get_ref('batch_size'),
              split_across_devices=True,
          ),
          _c('pipeline.NormalizeAudio', target_gain=0.2),
      ],
  )
  eval_dataset_config.split = 'train'
  eval_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  eval_dataset_config.dataset_directory = eval_dataset_dir
  return eval_dataset_config


def get_conformer_config(**kwargs) -> config_dict.ConfigDict:
  """Default conformer config."""
  conformer_config = config_dict.ConfigDict()
  conformer_config.model_dims = 768
  conformer_config.kernel_size = 32
  conformer_config.ff_activation = config_utils.object_config('nn.swish')
  conformer_config.ff_residual_weight = 0.5
  conformer_config.ffn_dim_multiplier = 4
  conformer_config.atten_num_heads = 8
  conformer_config.layer_order = 'mhsa_before_conv'
  conformer_config.dropout_prob = 0.0
  conformer_config.conv_residual_dropout = None
  conformer_config.atten_residual_dropout = None
  conformer_config.ffn_residual_dropout = None
  conformer_config.atten_dropout = None
  conformer_config.ffn_relu_dropout = None
  conformer_config.fflayer_weight_sharing = False
  conformer_config.num_blocks = 12
  conformer_config.skip_layer_norm = True
  conformer_config.update(**kwargs)
  return conformer_config


def get_early_fs_config(**kwargs) -> config_dict.ConfigDict:
  """Default early feature extractor config."""
  early_fs_config = config_dict.ConfigDict()
  early_fs_config.omit_earlyfs = False
  early_fs_config.dropout_prob = 0.0
  early_fs_config.activation = config_utils.object_config('nn.gelu')
  early_fs_config.num_frames = 500
  early_fs_config.deprecated_group_conv = False
  early_fs_config.update(**kwargs)
  return early_fs_config


def get_mask_config(**kwargs) -> config_dict.ConfigDict:
  """Default mask config."""
  mask_config = config_dict.ConfigDict()
  mask_config.mask_prob = 0.16
  mask_config.mask_length = 10
  mask_config.min_masks = 1
  mask_config.update(**kwargs)
  return mask_config


def get_classifier_config(**kwargs) -> config_dict.ConfigDict:
  """Default classifier config."""
  classifier_config = config_dict.ConfigDict()
  classifier_config.classify_from_all = True
  classifier_config.per_frame_predictions = True
  classifier_config.classify_pool_width = 3
  classifier_config.classify_stride = 3
  classifier_config.classify_features = 512
  classifier_config.reduction_type = 'AVG'
  classifier_config.update(**kwargs)
  return classifier_config


def get_quantizer_config(
    **kwargs,
) -> config_dict.ConfigDict:
  """Default quantizer config."""
  quantizer_config = config_dict.ConfigDict()
  quantizer_config.num_sections = 16
  quantizer_config.strategy = 'product_quantization'
  quantizer_config.use_entropy_quantizer = True
  quantizer_config.update(**kwargs)
  return quantizer_config


def get_base_quantizer_config(
    **kwargs,
) -> config_dict.ConfigDict:
  """Default base quantizer config."""
  base_quantizer_config = config_dict.ConfigDict()
  base_quantizer_config.num_centroids = 64
  base_quantizer_config.gamma = 2
  base_quantizer_config.init_scale = 0.1
  base_quantizer_config.update(**kwargs)
  return base_quantizer_config


def get_model_config(**kwargs) -> config_dict.ConfigDict:
  """Default model config."""
  model_config = config_dict.ConfigDict()
  model_config.final_dim = 64  # the dim to project *each feature section* (PQ)
  model_config.logit_temp = 0.1
  model_config.alpha = 1.0
  model_config.taxonomy_loss_weight = 0.0
  model_config.readout_points = [3, 4, 5, 6, 7]
  model_config.quantizer_points = (-2,)
  model_config.stop_gradient_earlyfs = False
  model_config.use_raw_audio = True
  model_config.update(**kwargs)
  return model_config
