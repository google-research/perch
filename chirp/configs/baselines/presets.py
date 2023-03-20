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

"""Presets for the baseline experiments."""

from chirp.config_utils import callable_config as _c
from ml_collections import config_dict


def get_base_config(**kwargs):
  """Creates the base config object.

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
  config.eval_window_stride_s = 2.5
  config.frame_rate_hz = 100
  config.num_channels = 160
  config.kernel_size = 2_048  # ~0.08 ms * 32,000 Hz
  config.batch_size = 256
  config.target_class_list = 'xenocanto'
  config.num_train_steps = 200_000
  config.tfds_data_dir = ''
  config.update(kwargs)
  return config


def _compute_input_shape(
    config: config_dict.ConfigDict, window_size_ref: config_dict.FieldReference
) -> tuple[int, int]:
  """Computes the models's input shape."""
  # As explained in chirp.models.frontent.STFT, the output of
  # chirp.data.pipeline.MelSpectrogram has shape [num_frames, num_channels], and
  # num_frames is computed as
  #
  #   (num_samples + stride - (kernel_size % 2)) // stride - correction,
  #
  # where correction is 1 if kernel_size is even and 0 otherwise.
  num_samples = window_size_ref * config.get_ref('sample_rate_hz')
  stride = config.get_ref('sample_rate_hz') // config.get_ref('frame_rate_hz')
  odd_kernel = config.get_ref('kernel_size') % 2
  return (
      (num_samples + stride - odd_kernel) // stride + (odd_kernel - 1),
      config.get_ref('num_channels'),
  )


def get_base_init_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  """Default init config."""
  init_config = config_dict.ConfigDict()
  init_config.input_shape = _compute_input_shape(
      config, config.get_ref('train_window_size_s')
  )
  init_config.learning_rate = 0.0001
  init_config.optimizer = _c(
      'optax.adam', learning_rate=init_config.get_ref('learning_rate')
  )
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


def get_base_train_dataset_config(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  return get_supervised_train_pipeline(
      config,
      filtering_df_paths=None,
      filter_by_complement=False,  # Unused because filtering_df_path=None.
      train_dataset_dir='bird_taxonomy/upstream_slice_peaked:1.4.0',
  )


def get_ablation_train_dataset_config(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  return get_supervised_train_pipeline(
      config,
      filtering_df_paths=None,
      filter_by_complement=True,
      train_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
  )


def get_base_eval_config(
    config: config_dict.ConfigDict, **kwargs
) -> config_dict.ConfigDict:
  eval_config = config_dict.ConfigDict()
  eval_config.num_train_steps = config.get_ref('num_train_steps')
  eval_config.tflite_export = False
  eval_config.input_shape = _compute_input_shape(
      config, config.get_ref('eval_window_size_s')
  )
  eval_config.update(**kwargs)
  return eval_config


def get_base_eval_dataset_config(
    config: config_dict.ConfigDict,
) -> dict[str, config_dict.ConfigDict]:
  return {
      'powdermill': get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,  # Unused because filtering_df_path=None.
          slice_method='strided_windows',
          slice_start=0.0,
          eval_dataset_dir='soundscapes/powdermill_full_length:1.3.0',
      ),
  }


def get_ablation_eval_dataset_config(
    config: config_dict.ConfigDict,
) -> dict[str, config_dict.ConfigDict]:
  return {
      'powdermill': get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,  # Unused because filtering_df_path=None.
          slice_method='strided_windows',
          slice_start=0.0,
          eval_dataset_dir='soundscapes/powdermill_full_length:1.3.0',
      ),
      'train_iid_subset': get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,
          slice_method='fixed',
          slice_start=0.5,
          eval_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      ),
      'test_iid': get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,
          slice_method='fixed',
          slice_start=0.5,
          eval_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      ),
      'test_label_shifted': get_supervised_eval_pipeline(
          config,
          filtering_df_paths=None,
          filter_by_complement=False,
          slice_method='fixed',
          slice_start=0.5,
          eval_dataset_dir='bird_taxonomy/slice_peaked:1.4.0',
      ),
  }


def _get_pipeline_ops(
    filtering_df_paths: list[str] | None,
    filter_by_complement: bool,
    shuffle: bool,
    target_class_list: str,
    mixup: bool,
    slice_method: str,
    slice_window_size: int,
    slice_window_stride: float,
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
  filtering_ops = []
  shuffle_op = mixup_op = repeat_op = None
  melspec_stride = sample_rate // melspec_frame_rate
  if filtering_df_paths:
    for filtering_df_path in filtering_df_paths:
      filtering_ops.append(
          _c(
              'pipeline.FilterByFeature',
              filtering_df_path=filtering_df_path,
              complement=filter_by_complement,
          )
      )
  if shuffle:
    shuffle_op = _c('pipeline.Shuffle', shuffle_buffer_size=512)
  if mixup:
    mixup_op = _c('pipeline.MixAudio', target_dist=(1.0, 0.5, 0.25, 0.25))
  if slice_method == 'random':
    slice_op = _c('pipeline.RandomSlice', window_size=slice_window_size)
    annotate_op = None
  elif slice_method == 'fixed':
    slice_op = _c(
        'pipeline.Slice',
        window_size=slice_window_size,
        start=slice_start,
    )
    annotate_op = None
  elif slice_method == 'strided_windows':
    slice_op = _c(
        'pipeline.ExtractStridedWindows',
        window_length_sec=slice_window_size,
        window_stride_sec=slice_window_stride,
    )
    annotate_op = _c(
        'pipeline.DenselyAnnotateWindows', drop_annotation_bounds=True
    )
  else:
    raise ValueError(f'unrecognized slice method: {slice_method}')
  if random_normalize:
    normalize_op = _c(
        'pipeline.RandomNormalizeAudio', min_gain=0.15, max_gain=0.25
    )
  else:
    normalize_op = _c('pipeline.NormalizeAudio', target_gain=0.2)
  if repeat:
    repeat_op = _c('pipeline.Repeat')

  ops = filtering_ops + [
      shuffle_op,
      _c('pipeline.OnlyJaxTypes'),
      slice_op,
      annotate_op,
      # NOTE: pipeline.ConvertBirdTaxonomyLabels comes *after* the slicing and
      # annotation ops, as the pipeline.DenselyAnnotateWindows op used when
      # slice_method == 'strided_windows' expects labels to be sequences of
      # integers rather than multi-hot encoded vectors.
      _c(
          'pipeline.ConvertBirdTaxonomyLabels',
          source_namespace='ebird2021',
          target_class_list=target_class_list,
          add_taxonomic_labels=False,
      ),
      mixup_op,
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
    filtering_df_paths: list[str] | None,
    filter_by_complement: bool,
    train_dataset_dir: str,
) -> config_dict.ConfigDict:
  """Creates the supervised training data pipeline."""
  if train_dataset_dir not in (
      'bird_taxonomy/upstream_slice_peaked:1.4.0',
      'bird_taxonomy/slice_peaked:1.4.0',
  ):
    raise ValueError('we assume training on XC')
  train_dataset_config = config_dict.ConfigDict()
  train_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=_get_pipeline_ops(
          filtering_df_paths=filtering_df_paths,
          filter_by_complement=filter_by_complement,
          shuffle=True,
          target_class_list=config.get_ref('target_class_list'),
          mixup=True,
          slice_method='random',
          slice_window_size=config.get_ref('train_window_size_s'),
          slice_window_stride=0.0,  # Unused because slice_method=random'.
          slice_start=0.0,  # Unused because slice_method='random'.
          random_normalize=True,
          melspec_num_channels=config.get_ref('num_channels'),
          melspec_frame_rate=config.get_ref('frame_rate_hz'),
          melspec_kernel_size=config.get_ref('kernel_size'),
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
    filtering_df_paths: list[str] | None,
    filter_by_complement: bool,
    slice_method: str,
    slice_start: float,
    eval_dataset_dir: str,
) -> config_dict.ConfigDict:
  """Creates an eval data pipeline."""
  eval_dataset_config = config_dict.ConfigDict()
  eval_dataset_config.pipeline = _c(
      'pipeline.Pipeline',
      ops=_get_pipeline_ops(
          filtering_df_paths=filtering_df_paths,
          filter_by_complement=filter_by_complement,
          shuffle=False,
          target_class_list=config.get_ref('target_class_list'),
          mixup=False,
          slice_method=slice_method,
          slice_window_size=config.get_ref('eval_window_size_s'),
          slice_window_stride=config.get_ref('eval_window_stride_s'),
          slice_start=slice_start,
          random_normalize=False,
          melspec_num_channels=config.get_ref('num_channels'),
          melspec_frame_rate=config.get_ref('frame_rate_hz'),
          melspec_kernel_size=config.get_ref('kernel_size'),
          sample_rate=config.get_ref('sample_rate_hz'),
          batch_size=config.get_ref('batch_size'),
          repeat=False,
      ),
  )
  eval_dataset_config.split = 'train'
  eval_dataset_config.tfds_data_dir = config.get_ref('tfds_data_dir')
  eval_dataset_config.dataset_directory = eval_dataset_dir
  return eval_dataset_config
