# coding=utf-8
# Copyright 2023 The Chirp Authors.
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
evaluation is performed over a pretrained EfficientNet model.
"""

import itertools

from typing import Sequence
from chirp import config_utils
from ml_collections import config_dict

_callable_config = config_utils.callable_config
_object_config = config_utils.object_config

_TFDS_DATA_DIR = None


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
          'dataset_name': 'xc_artificially_rare_class_reps_v2',
          'to_crop': True,
          'tfds_name': 'bird_taxonomy/upstream_ar_only_slice_peaked:2.*.*',
      },
      {
          'dataset_name': 'xc_downstream_class_reps',
          'to_crop': True,
          'tfds_name': 'bird_taxonomy/downstream_slice_peaked:2.*.*',
      },
      # The `xc_downstream` dataset includes feasible artificially rare species
      # and downstream species with which to construct search corpora.
      {
          'dataset_name': 'xc_downstream',
          'to_crop': False,
          'tfds_name': 'bird_taxonomy/downstream_full_length:2.*.*',
      },
      {
          'dataset_name': 'soundscapes_ssw',
          'to_crop': False,
          'tfds_name': 'soundscapes/ssw_full_length',
      },
      {
          'dataset_name': 'soundscapes_coffee_farms',
          'to_crop': False,
          'tfds_name': 'soundscapes/coffee_farms_full_length',
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

    if dataset_description['to_crop']:
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
  config.eval_set_specifications = {}
  for corpus_type, location in itertools.product(
      ('xc_fg', 'xc_bg', 'soundscapes'), ('ssw', 'coffee_farms', 'hawaii')
  ):
    # SSW species are "artificially rare" (a limited number of examples were
    # included during upstream training). If provided, we use the singular
    # learned vector representation from upstream training during search.
    # Otherwise, we use all available upstream recordings.
    if location == 'ssw':
      config.eval_set_specifications[f'artificially_rare_{corpus_type}'] = (
          _callable_config(
              'eval_lib.EvalSetSpecification.v2_specification',
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
            'eval_lib.EvalSetSpecification.v2_specification',
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

  # The following two fields should be populated by the user in an eval config.
  # Each should point to a local function, callable, or one of the provided
  # functions in

  # google-research/chirp/eval/eval_lib.py.
  config.create_species_query = None

  # Determines the ordering of search results for use in average-precision based
  # metrics. For similarity-based metrics, set sort_descending to True. For
  # distance-based metrics, set this to False (for ascending ordering).
  config.sort_descending = None

  return config
