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

"""Configuration to run the logistic regression baseline."""
from chirp import config_utils
from chirp.configs.debugging import presets
from ml_collections import config_dict

_c = config_utils.callable_config
_KERNEL_SIZE = 2_048  # ~0.08 ms * 32,000 Hz


def get_pipeline_ops(
    filtering_df_path: str | None,
    filter_by_complement: bool,
    shuffle: bool,
    target_class_list: str,
    mixup: bool,
    random_slice: bool,
    slice_window_size: int,
    slice_start: float,
    random_normalize: bool,
    melspec_num_channels: int,
    melspec_frame_rate: int,
    melspec_kernel_size: int,
    sample_rate: int,
    batch_size: int,
    repeat: bool,
) -> list[config_dict.ConfigDict]:
  """Creates the pipeline ops."""
  filtering_op = shuffle_op = mixup_op = repeat_op = None
  melspec_stride = sample_rate // melspec_frame_rate
  if filtering_df_path:
    filtering_op = _c(
        'pipeline.FilterByFeature',
        filtering_df_path=filtering_df_path,
        complement=filter_by_complement,
    )
  if shuffle:
    shuffle_op = _c('pipeline.Shuffle', shuffle_buffer_size=512)
  if mixup:
    mixup_op = _c('pipeline.MixAudio', target_dist=(1.0, 0.5, 0.25, 0.25))
  if random_slice:
    slice_op = _c('pipeline.RandomSlice', window_size=slice_window_size)
  else:
    slice_op = _c(
        'pipeline.Slice',
        window_size=slice_window_size,
        start=slice_start,
    )
  if random_normalize:
    normalize_op = _c(
        'pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25
    )
  else:
    normalize_op = _c('pipeline.NormalizeAudio', target_gain=0.2)
  if repeat:
    repeat_op = _c('pipeline.Repeat')

  ops = [
      filtering_op,
      shuffle_op,
      _c('pipeline.OnlyJaxTypes'),
      _c(
          'pipeline.ConvertBirdTaxonomyLabels',
          source_namespace='ebird2021',
          target_class_list=target_class_list,
          add_taxonomic_labels=False,
      ),
      mixup_op,
      slice_op,
      normalize_op,
      _c(
          'pipeline.MelSpectrogram',
          features=melspec_num_channels,
          stride=melspec_stride,
          kernel_size=melspec_kernel_size,
          sample_rate=sample_rate,
          freq_range=(60, 10_000),
          # Settings from PCEN: Why and how
          scaling_config=_c(
              'frontend.PCENScalingConfig',
              # Disable convolutional approximation
              conv_width=0,
              # Solution to 2*pi*tau/T = arccos(1 - s^2/(2 * (1 - s)))
              # (prop III.1) for tau = 1.5 ms and T = 60 ms
              smoothing_coef=0.145,
              gain=0.8,
              bias=10.0,
              root=4.0,
          ),
      ),
      _c(
          'pipeline.Batch',
          batch_size=batch_size,
          split_across_devices=True,
      ),
      repeat_op,
  ]
  return [op for op in ops if op is not None]


def get_supervised_train_pipeline(
    config: config_dict.ConfigDict,
    filtering_df_path: str | None,
    filter_by_complement: bool,
    train_dataset_dir: str,
) -> config_dict.ConfigDict:
  """Creates the supervised training data pipeline."""
  if train_dataset_dir != 'bird_taxonomy/slice_peaked:1.4.0':
    raise ValueError('we assume training on XC')
  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=get_pipeline_ops(
          filtering_df_path=filtering_df_path,
          filter_by_complement=filter_by_complement,
          shuffle=True,
          target_class_list=config.get_ref('target_class_list'),
          mixup=True,
          random_slice=True,
          slice_window_size=config.get_ref('train_window_size_s'),
          slice_start=0.0,  # Unused because random_slice = True.
          random_normalize=True,
          melspec_num_channels=config.get_ref('num_channels'),
          melspec_frame_rate=config.get_ref('frame_rate_hz'),
          melspec_kernel_size=_KERNEL_SIZE,
          sample_rate=config.get_ref('sample_rate_hz'),
          batch_size=config.get_ref('batch_size'),
          repeat=True,
      ),
  )
  train_dataset_config.split = 'train'
  train_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  train_dataset_config.dataset_directory = train_dataset_dir
  return train_dataset_config


def get_supervised_eval_pipeline(
    config: config_dict.ConfigDict,
    filtering_df_path: str | None,
    filter_by_complement: bool,
    slice_start: float,
    eval_dataset_dir: str,
) -> config_dict.ConfigDict:
  """Creates an eval data pipeline."""
  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=get_pipeline_ops(
          filtering_df_path=filtering_df_path,
          filter_by_complement=filter_by_complement,
          shuffle=False,
          target_class_list=config.get_ref('target_class_list'),
          mixup=False,
          random_slice=False,
          slice_window_size=config.get_ref('train_window_size_s'),
          slice_start=slice_start,
          random_normalize=False,
          melspec_num_channels=config.get_ref('num_channels'),
          melspec_frame_rate=config.get_ref('frame_rate_hz'),
          melspec_kernel_size=_KERNEL_SIZE,
          sample_rate=config.get_ref('sample_rate_hz'),
          batch_size=config.get_ref('batch_size'),
          repeat=False,
      ),
  )
  eval_dataset_config.split = 'train'
  eval_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  eval_dataset_config.dataset_directory = eval_dataset_dir
  return eval_dataset_config


def get_config() -> config_dict.ConfigDict:
  """Creates the configuration dictionary for training and evaluation."""
  config = presets.get_base_config()

  # Configure the data
  config.train_dataset_config = get_supervised_train_pipeline(
      config,
      filtering_df_path=None,
      filter_by_complement=True,
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
  )
  config.eval_dataset_config = {
      'caples': get_supervised_eval_pipeline(
          config,
          filtering_df_path=None,
          filter_by_complement=False,
          slice_start=0.0,
          eval_dataset_dir='soundscapes/caples:1.1.0',
      ),
      'xc_train_subset': get_supervised_eval_pipeline(
          config,
          filtering_df_path=None,
          filter_by_complement=False,
          slice_start=0.5,
          eval_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      ),
      'xc_test': get_supervised_eval_pipeline(
          config,
          filtering_df_path=None,
          filter_by_complement=False,
          slice_start=0.5,
          eval_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      ),
  }
  # Configure the experiment setup
  config.init_config = presets.get_base_init_config(config)
  config.init_config.optimizer = _c(
      'optax.adam', learning_rate=config.init_config.get_ref('learning_rate')
  )

  encoder_config = config_dict.ConfigDict()
  encoder_config.aggregation = 'beans'
  encoder_config.compute_mfccs = True
  encoder_config.num_mfccs = 20
  config.encoder_config = encoder_config

  model_config = config_dict.ConfigDict()
  model_config.encoder = _c(
      'handcrafted_features.HandcraftedFeatures',
      compute_mfccs=encoder_config.get_ref('compute_mfccs'),
      num_mfccs=encoder_config.get_ref('num_mfccs'),
      aggregation=encoder_config.get_ref('aggregation'),
      window_size=10,
      window_stride=10,
  )
  model_config.taxonomy_loss_weight = 0.0
  model_config.frontend = None
  config.init_config.model_config = model_config
  # Configure the training loop
  num_train = config.get_ref('train_window_size_s') * config.get_ref(
      'sample_rate_hz'
  )
  num_eval = config.get_ref('eval_window_size_s') * config.get_ref(
      'sample_rate_hz'
  )
  stride = config.get_ref('sample_rate_hz') // config.get_ref(
      'frame_rate_hz'
  )
  # As explained in chirp.models.frontent.STFT, the output of
  # chirp.data.pipeline.MelSpectrogram has shape [num_frames, num_channels], and
  # num_frames is computed as
  #
  #   (num_samples + stride - (kernel_size % 2)) // stride - correction,
  #
  # where correction is 1 if kernel_size is even and 0 otherwise.
  odd_kernel = _KERNEL_SIZE % 2
  num_train_frames = (
      (num_train + stride - odd_kernel) // stride + odd_kernel - 1
  )
  num_eval_frames = (num_eval + stride - odd_kernel) // stride + odd_kernel - 1
  config.init_config.input_shape = (
      num_train_frames,
      config.get_ref('num_channels'),
  )
  config.train_config = presets.get_base_train_config(config)
  config.eval_config = presets.get_base_eval_config(
      config,
      input_shape=(num_eval_frames, config.get_ref('num_channels')),
  )
  return config


def get_hyper(hyper):
  """Defines the hyperparameter sweep."""
  encoder_hypers = hyper.zipit([
      hyper.sweep(
          'config.encoder_config.aggregation',
          ['beans', 'flatten', 'avg_pool'],
      ),
      hyper.sweep(
          'config.encoder_config.compute_mfccs',
          [True, True, False],
      ),
  ])
  optimizer_hypers = hyper.sweep(
      'config.init_config.learning_rate',
      hyper.discrete([1e-3, 1e-2, 1e-1, 1e0]),
  )
  return hyper.product([encoder_hypers, optimizer_hypers])
