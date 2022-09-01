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

"""Configuration for evaluating using the MVP protocol."""

from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config

_TFDS_DATA_DIR = None


def get_config() -> config_dict.ConfigDict:
  """Creates a configuration dictionary for the MVP evaluation protocol.

  The MVP protocol evaluates on artificially rare Sapsucker Woods (SSW) species
  as well as on held-out Colombia and Hawaii species.

  Returns:
    The configuration dictionary for the MVP evaluation protocol.
  """
  config = config_dict.ConfigDict()
  config.tfds_data_dir = config_dict.FieldReference(_TFDS_DATA_DIR)

  # Xeno-Canto's slice_peaked variants contain 6-second audio segments that are
  # randomly cropped to 5-second segments during training. At evaluation, we
  # center-crop them down to 5-second segments. Soundscapes' audio segments are
  # already 5-seconds long and do not need any cropping.
  xc_window_size_seconds = 5
  xc_slice_start = 0.5
  # The audio is normalized to a target gain of 0.2.
  target_gain = 0.2

  required_datasets = (
      {
          'dataset_name': 'xc_downstream',
          'is_xc': True,
          'tfds_name': 'bird_taxonomy/downstream_slice_peaked'
      },
      {
          'dataset_name': 'birdclef_ssw',
          'is_xc': False,
          'tfds_name': 'soundscapes/birdclef2019_ssw'
      },
      {
          'dataset_name': 'birdclef_colombia',
          'is_xc': False,
          'tfds_name': 'soundscapes/birdclef2019_colombia'
      },
      # TODO(vdumoulin): add Hawaii once b/243792097 is fixed.
  )

  dataset_configs = {}
  for dataset_description in required_datasets:
    dataset_config = config_dict.ConfigDict()
    dataset_config.tfds_name = dataset_description['tfds_name']
    dataset_config.tfds_data_dir = config.tfds_data_dir

    ops = [
        _c('pipeline.OnlyJaxTypes'),
        _c('pipeline.NormalizeAudio', target_gain=target_gain)
    ]
    # Xeno-Canto data needs to be cropped before normalizing the audio.
    if dataset_description['is_xc']:
      slice_op = _c(
          'pipeline.Slice',
          window_size=xc_window_size_seconds,
          start=xc_slice_start)
      ops.insert(1, slice_op)

    dataset_config.pipeline = _c('pipeline.Pipeline', ops=ops)
    dataset_config.split = 'train'
    dataset_configs[dataset_description['dataset_name']] = dataset_config

  config.dataset_configs = dataset_configs

  return config
