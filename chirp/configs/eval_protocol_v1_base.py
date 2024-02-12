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

"""Base configuration for model evaluation using the v1 protocol."""

import itertools

from chirp import config_utils
from ml_collections import config_dict

_callable_config = config_utils.callable_config
_object_config = config_utils.object_config

_TFDS_DATA_DIR = None


def _crop_if_slice_peaked(to_crop: bool, **kwargs):
  return [_callable_config('pipeline.Slice', **kwargs)] if to_crop else []


def _melspec_if_baseline(config_string: str, **kwargs):
  return (
      [_callable_config('pipeline.MelSpectrogram', **kwargs)]
      if config_string == 'baseline'
      else []
  )


def get_config() -> config_dict.ConfigDict:
  """Creates a base configuration dictionary for the v1 evaluation protocol.

  The v1 protocol evaluates on artificially rare Sapsucker Woods (SSW) species
  and on held-out Colombia and Hawaii species.

  Returns:
    The base configuration dictionary for the v1 evaluation protocol.
  """
  config = config_dict.ConfigDict()

  tfds_data_dir = config_dict.FieldReference(_TFDS_DATA_DIR)
  config.tfds_data_dir = tfds_data_dir

  # The PRNG seed controls the random subsampling of class representatives down
  # to the right number of when forming eval sets.
  config.rng_seed = 1234
  config.write_results_dir = '/tmp/'
  config.batch_size = 16

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
          'dataset_name': 'xc_artificially_rare',
          'to_crop': True,
          'tfds_name': 'bird_taxonomy/upstream_ar_only_slice_peaked:1.*.*',
      },
      {
          'dataset_name': 'xc_downstream',
          'to_crop': True,
          'tfds_name': 'bird_taxonomy/downstream_slice_peaked:1.*.*',
      },
      {
          'dataset_name': 'birdclef_ssw',
          'to_crop': False,
          'tfds_name': 'soundscapes/ssw',
      },
      {
          'dataset_name': 'birdclef_colombia',
          'to_crop': False,
          'tfds_name': 'soundscapes/birdclef2019_colombia',
      },
  )

  dataset_configs = {}
  for dataset_description in required_datasets:
    dataset_config = config_dict.ConfigDict()
    dataset_config.tfds_name = dataset_description['tfds_name']
    dataset_config.tfds_data_dir = tfds_data_dir

    ops = [
        _callable_config(
            'pipeline.OnlyKeep',
            names=[
                'audio',
                'label',
                'bg_labels',
                'recording_id',
                'segment_id',
            ],
        ),
        # Xeno-Canto data needs to be cropped before normalizing the audio.
        _crop_if_slice_peaked(
            dataset_description['to_crop'],
            window_size=xc_window_size_seconds,
            start=xc_slice_start,
        ),
        _callable_config('pipeline.NormalizeAudio', target_gain=target_gain),
        _callable_config('pipeline.LabelsToString'),
    ]

    dataset_config.pipeline = _callable_config(
        'pipeline.Pipeline', ops=ops, deterministic=True
    )
    dataset_config.split = 'train'
    dataset_configs[dataset_description['dataset_name']] = dataset_config

  config.dataset_configs = dataset_configs

  # Build all eval set specifications.
  config.eval_set_specifications = {}
  for corpus_type, location in itertools.product(
      ('xc_fg', 'xc_bg', 'birdclef'), ('ssw', 'colombia', 'hawaii')
  ):
    # SSW species are "artificially rare" (a limited number of examples were
    # included during upstream training). If provided, we use the singular
    # learned vector representation from upstream training during search.
    # Otherwise, we use all available upstream recordings.
    if location == 'ssw':
      config.eval_set_specifications[f'artificially_rare_{corpus_type}'] = (
          _callable_config(
              'eval_lib.EvalSetSpecification.v1_specification',
              location=location,
              corpus_type=corpus_type,
              num_representatives_per_class=-1,
          )
      )
    # For downstream species, we sweep over {1, 2, 4, 8, 16} representatives
    # per class, and in each case we resample the collection of class
    # representatives 5 times to get confidence intervals on the metrics.
    else:
      for k, seed in itertools.product((1, 2, 4, 8, 16), range(1, 6)):
        config.eval_set_specifications[
            f'{location}_{corpus_type}_{k}_seed{seed}'
        ] = _callable_config(
            'eval_lib.EvalSetSpecification.v1_specification',
            location=location,
            corpus_type=corpus_type,
            num_representatives_per_class=k,
        )

  config.debug = config_dict.ConfigDict()
  # Path to the embedded dataset cache. If set, the embedded dataset will be
  # cached at that path and used upon subsequent runs without recomputing the
  # embeddings.
  #
  # **WARNING**: only use to speed up debugging. When the path is set and a
  # cache, already exists, the model callback will be ignored. No effect will
  # occur if there are updates to the model without updating the cache path
  # (i.e. metrics will be computed with respect to a previous model callback's
  # embeddings).
  config.debug.embedded_dataset_cache_path = ''

  # The following two fields should be populated by the user in an eval config,
  # and each point to a local function, callable, or to one of the functions
  # provided in

  # google-research/perch/eval/eval_lib.py.
  config.create_species_query = None

  # Determines the ordering of search results for use in average-precision based
  # metrics. For similarity-based metrics, set sort_descending to True. For
  # distance-based metrics, set this to False (for ascending ordering).
  config.sort_descending = None

  return config
