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

"""Base config for model evaluation using the v2 protocol.

This config sets up model evaluation using the generalization evaluation
framework over windowed and densely annotated examples without applying any
aggregation of scored search results at the recording or annotation level. The
evaluation is performed over the specified pre-trained model.
"""

import itertools
from typing import Dict, Sequence
from chirp import config_utils
from ml_collections import config_dict

_callable_config = config_utils.callable_config
_object_config = config_utils.object_config

_TFDS_DATA_DIR = None
_EVAL_REGIONS = (
    'ssw',
    'coffee_farms',
    'hawaii',
    'high_sierras',
    'sierras_kahl',  # Sierra Nevada region
    'peru',
)
_CORPUS_TYPES = ('xc_fg', 'xc_bg', 'soundscapes')
_NUM_REPS = (1, 2, 4, 8, 16)
_SEEDS = (1, 2, 3, 4, 5)


def build_eval_set_specs() -> Dict[str, config_dict.ConfigDict]:
  """Build EvalSetSpecifications with which to construct all eval datasets.

  In v2, a varied number of class representatives will be used to produce model
  embeddings to form the basis for species queries during eval_lib.search(). The
  representatives are resampled len(_SEEDS) times to be able to compute metric
  confidence intervals.

  Returns:
    A mapping of eval set specifier to a ConfigDict containing the unparsed
    EvalSetSpecification configs.
  """

  eval_set_specifications = {}

  for corpus_type, location in itertools.product(_CORPUS_TYPES, _EVAL_REGIONS):
    for k, seed in itertools.product(_NUM_REPS, _SEEDS):
      eval_set_specifications[f'{location}_{corpus_type}_{k}_seed{seed}'] = (
          _callable_config(
              'eval_lib.EvalSetSpecification.v2_specification',
              location=location,
              corpus_type=corpus_type,
              num_representatives_per_class=k,
          )
      )

  return eval_set_specifications


def get_config(
    data_ops: Sequence[config_dict.ConfigDict] | None = None,
) -> config_dict.ConfigDict:
  """Creates a base configuration dictionary for the v2 evaluation protocol.

  The v2 protocol evaluates on artificially rare Sapsucker Woods (SSW) species
  and on held-out Colombia and Hawaii species.

  Args:
    data_ops: An optional sequence of additional pipeline preprocessing data ops
      to add to the default configuration.

  Returns:
    The base configuration dictionary for the v2 evaluation protocol.
  """
  # If no additional data pipeline ops are passed, update to empty list for
  # downstream concatenation type matching.
  if not data_ops:
    data_ops = []

  config = config_dict.ConfigDict()

  tfds_data_dir = config_dict.FieldReference(_TFDS_DATA_DIR)
  config.tfds_data_dir = tfds_data_dir

  # The PRNG seed controls the random subsampling of class representatives down
  # to the right number of when forming eval sets.
  config.rng_seed = 1234
  config.write_results_dir = '/tmp/'
  config.batch_size = 1024

  # Xeno-Canto's slice_peaked variants contain 6-second audio segments that are
  # randomly cropped to 5-second segments during training. At evaluation, we
  # center-crop them down to 5-second segments. Soundscapes' audio segments are
  # already 5-seconds long and do not need any cropping.
  xc_window_size_seconds = 5
  xc_slice_start = 0.5

  # Hyperparameters for the v2 evaluation which uses strided windowing and
  # dense annotation.
  config.window_length_sec = 5
  config.window_stride_sec = 2.5
  config.overlap_threshold_sec = None

  required_datasets = (
      {
          'dataset_name': 'xc_class_reps',
          'tfds_name': 'bird_taxonomy/class_representatives_slice_peaked:2.*.*',
      },
      # The `xc_downstream` dataset includes feasible artificially rare species
      # and downstream species with which to construct search corpora.
      {
          'dataset_name': 'xc_downstream',
          'tfds_name': 'bird_taxonomy/downstream_full_length:2.*.*',
      },
      {
          'dataset_name': 'soundscapes_ssw',
          'tfds_name': 'soundscapes/ssw_full_length',
      },
      {
          'dataset_name': 'soundscapes_coffee_farms',
          'tfds_name': 'soundscapes/coffee_farms_full_length',
      },
      {
          'dataset_name': 'soundscapes_hawaii',
          'tfds_name': 'soundscapes/hawaii_full_length',
      },
      {
          'dataset_name': 'soundscapes_peru',
          'tfds_name': 'soundscapes/peru_full_length',
      },
      {
          'dataset_name': 'soundscapes_high_sierras',
          'tfds_name': 'soundscapes/high_sierras_full_length',
      },
      {
          'dataset_name': 'soundscapes_sierras_kahl',
          'tfds_name': 'soundscapes/sierras_kahl_full_length',
      },
  )

  # Construct Pipelines to process slice-peaked and full-length datasets.
  # Xeno-Canto class representative data needs to be cropped down to 5sec before
  # normalizing the audio.
  slice_peaked_pipeline_ops = [
      _callable_config(
          'pipeline.Slice',
          window_size=xc_window_size_seconds,
          start=xc_slice_start,
      ),
      _callable_config(
          'pipeline.OnlyKeep',
          names=[
              'audio',
              'label',
              'bg_labels',
              'recording_id',
              'segment_id',
              'segment_start',
              'segment_end',
          ],
      ),
      _callable_config('pipeline.LabelsToString'),
  ]

  # Full-length Xeno-Canto recordings are processed to extract strided windows.
  # Each strided window receives the recording-level annotations. (Note that for
  # this dataset, we do not have human segment-level annotations, so we do not
  # follow the same process as with soundscapes downstream full-length
  # recordings.)
  full_length_xc_pipeline_ops = [
      _callable_config(
          'pipeline.ExtractStridedWindows',
          window_length_sec=config.window_length_sec,
          window_stride_sec=config.window_stride_sec,
          pad_end=True,
      ),
      _callable_config(
          'pipeline.OnlyKeep',
          names=[
              'audio',
              'label',
              'bg_labels',
              'recording_id',
              'segment_id',
              'segment_start',
              'segment_end',
          ],
      ),
      # NOTE: this pipeline operation should be applied after window extraction,
      # dense annotation, and the OnlyKeep operation. This op turns a sequence
      # of labels into a single space-separated string of species codes;
      # the previous ops assume that labels are sequences of int IDs.
      _callable_config('pipeline.LabelsToString'),
  ]

  # Full-length recordings are used to construct the search corpora data for
  # soundscapes. Slices are constructed using strided windowing and dense
  # annotation.
  full_length_soundscapes_pipeline_ops = [
      _callable_config(
          'pipeline.ExtractStridedWindows',
          window_length_sec=config.window_length_sec,
          window_stride_sec=config.window_stride_sec,
          pad_end=True,
      ),
      _callable_config(
          'pipeline.DenselyAnnotateWindows',
          overlap_threshold_sec=config.overlap_threshold_sec,
          drop_annotation_bounds=True,
      ),
      _callable_config(
          'pipeline.OnlyKeep',
          names=[
              'audio',
              'label',
              'bg_labels',
              'recording_id',
              'segment_id',
              'segment_start',
              'segment_end',
          ],
      ),
      # NOTE: this pipeline operation should be applied at the very end, as it
      # turns a sequence of labels into a single space-separated string of
      # species codes. Previous ops in the pipeline assume that labels are
      # sequences of integer IDs.
      _callable_config('pipeline.LabelsToString'),
  ]

  dataset_configs = {}
  for dataset_description in required_datasets:
    dataset_config = config_dict.ConfigDict()
    dataset_config.tfds_name = dataset_description['tfds_name']
    dataset_config.tfds_data_dir = tfds_data_dir

    if dataset_description['dataset_name'] == 'xc_class_reps':
      ops = slice_peaked_pipeline_ops + data_ops
    elif dataset_description['dataset_name'] == 'xc_downstream':
      ops = full_length_xc_pipeline_ops + data_ops
    else:
      ops = full_length_soundscapes_pipeline_ops + data_ops

    dataset_config.pipeline = _callable_config(
        'pipeline.Pipeline', ops=ops, deterministic=True
    )
    dataset_config.split = 'train'
    dataset_configs[dataset_description['dataset_name']] = dataset_config

  config.dataset_configs = dataset_configs

  # Build all eval set specifications.
  config.eval_set_specifications = build_eval_set_specs()

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

  # The following two fields should be populated by the user in an eval config.
  # Each should point to a local function, callable, or one of the provided
  # functions in

  # google-research/perch/eval/eval_lib.py.
  config.create_species_query = None

  # Determines the ordering of search results for use in average-precision based
  # metrics. For similarity-based metrics, set sort_descending to True. For
  # distance-based metrics, set this to False (for ascending ordering).
  config.sort_descending = None

  return config
