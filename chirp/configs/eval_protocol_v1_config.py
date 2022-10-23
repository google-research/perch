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

"""Configuration for evaluating using the v1 protocol."""

import itertools

from chirp import config_utils
from chirp.configs import baseline_attention
from ml_collections import config_dict

_c = config_utils.callable_config
_object_config = config_utils.object_config

_TFDS_DATA_DIR = None


def get_config() -> config_dict.ConfigDict:
  """Creates a configuration dictionary for the evaluation protocol v1.

  The v1 protocol evaluates on artificially rare Sapsucker Woods (SSW) species
  as well as on held-out Colombia and Hawaii species.

  Returns:
    The configuration dictionary for the v1 evaluation protocol.
  """
  config = config_dict.ConfigDict()
  baseline_attention_config = baseline_attention.get_config()
  tfds_data_dir = config_dict.FieldReference(_TFDS_DATA_DIR)
  config.tfds_data_dir = tfds_data_dir
  # The model_callback is expected to be a Callable[[np.ndarray], np.ndarray].
  model_checkpoint_path = config_dict.FieldReference('')
  config.model_checkpoint_path = model_checkpoint_path
  config.model_callback = _c(
      'eval_lib.FlaxCheckpointCallback',
      init_config=baseline_attention_config.init_config,
      workdir=model_checkpoint_path)
  config.batch_size = 16
  # The PRNG seed controls the random subsampling of class representatives down
  # to the right number of when forming eval sets.
  config.rng_seed = 1234

  # TODO(bringingjoy): extend create_species_query to support returning multiple
  # queries for a given eval species.
  config.create_species_query = _object_config('eval_lib.create_averaged_query')
  config.score_search = _object_config('eval_lib.cosine_similarity')
  config.score_search_ordering = 'high'
  # TODO(hamer): consider enforcing similarity ordering assumption for the user
  # in place of adding an ordering flag (to be passed to ../model/metric).
  # TODO(hamer): determine how to structure paths for model evaluation results.
  config.write_results_dir = '/tmp/'

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
          'is_xc': True,
          'tfds_name': 'bird_taxonomy/upstream_ar_only_slice_peaked'
      },
      {
          'dataset_name': 'xc_downstream',
          'is_xc': True,
          'tfds_name': 'bird_taxonomy/downstream_slice_peaked'
      },
      {
          'dataset_name': 'birdclef_ssw',
          'is_xc': False,
          'tfds_name': 'soundscapes/ssw'
      },
      {
          'dataset_name': 'birdclef_colombia',
          'is_xc': False,
          'tfds_name': 'soundscapes/birdclef2019_colombia'
      },
  )

  dataset_configs = {}
  for dataset_description in required_datasets:
    dataset_config = config_dict.ConfigDict()
    dataset_config.tfds_name = dataset_description['tfds_name']
    dataset_config.tfds_data_dir = tfds_data_dir

    ops = [
        _c('pipeline.OnlyKeep',
           names=['audio', 'label', 'bg_labels', 'recording_id', 'segment_id']),
        _c('pipeline.NormalizeAudio', target_gain=target_gain),
        baseline_attention_config.eval_dataset_config.pipeline.__config.ops[-1],  # pylint: disable=protected-access
        _c('pipeline.LabelsToString')
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

  # Build all eval set specifications.
  config.eval_set_specifications = {}
  for corpus_type, location in itertools.product(('xc_fg', 'xc_bg', 'birdclef'),
                                                 ('ssw', 'colombia', 'hawaii')):
    # SSW species are "artificially rare" (a limited number of examples were
    # included during upstream training). We use the singular learned vector
    # representation from upstream training during search.
    if location == 'ssw':
      config.eval_set_specifications[f'artificially_rare_{corpus_type}'] = _c(
          'eval_lib.EvalSetSpecification.v1_specification',
          location=location,
          corpus_type=corpus_type,
          num_representatives_per_class=-1)
    # For downstream species, we sweep over {1, 2, 4, 8, 16} representatives
    # per class, and in each case we resample the collection of class
    # representatives 5 times to get confidence intervals on the metrics.
    else:
      for k, seed in itertools.product((1, 2, 4, 8, 16), range(1, 6)):
        config.eval_set_specifications[
            f'{location}_{corpus_type}_{k}_seed{seed}'] = _c(
                'eval_lib.EvalSetSpecification.v1_specification',
                location=location,
                corpus_type=corpus_type,
                num_representatives_per_class=k)

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

  return config
