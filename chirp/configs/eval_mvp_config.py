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

import itertools

from chirp import config_utils
from chirp.eval import eval_lib
from chirp.taxonomy import namespace_db
from ml_collections import config_dict

_c = config_utils.callable_config

_TFDS_DATA_DIR = None


def _make_eval_set_config(
    location, corpus_type,
    num_representatives_per_class) -> eval_lib.EvalSetSpecification:
  """Instantiates an eval MVP EvalSetSpecification.

  Args:
    location: Geographical location in {'ssw', 'colombia', 'hawaii'}.
    corpus_type: Corpus type in {'xc_fg', 'xc_bg', 'birdclef'}.
    num_representatives_per_class: Number of class representatives to sample.

  Returns:
    The EvalSetSpecification.
  """
  downstream_class_names = namespace_db.load_db(
  ).class_lists['downstream_species'].classes
  # Some species are present in Colombia and Hawai'i but are excluded from the
  # downstream species because of their conservation status.
  class_names = {
      'ssw':
          namespace_db.load_db().class_lists['artificially_rare_species']
          .classes,
      'colombia': [
          c for c in
          namespace_db.load_db().class_lists['birdclef2019_colombia'].classes
          if c in downstream_class_names
      ],
      'hawaii': [
          c for c in namespace_db.load_db().class_lists['hawaii'].classes
          if c in downstream_class_names
      ],
  }[location]

  # The name of the dataset to draw embeddings from to form the corpus.
  dataset_name = (f'birdclef_{location}'
                  if corpus_type == 'birdclef' else 'xc_downstream')
  has_dataset_name = lambda df: df['dataset_name'] == dataset_name
  # We only include in the search corpus the embeddings which have *some*
  # annotation (either foreground or background) of *some* class in
  # `class_names`. The 'label' and 'bg_labels' features are represented as
  # strings of space-separated species codes.
  has_some_fg_annotation = (
      # `'|'.join(class_names)` is a regex which matches *any* class in
      # `class_names`.
      lambda df: df['label'].str.contains('|'.join(class_names)))
  has_some_bg_annotation = (
      lambda df: df['bg_labels'].str.contains('|'.join(class_names)))
  has_some_annotation = (
      lambda df: has_some_fg_annotation(df) | has_some_bg_annotation(df))

  return eval_lib.EvalSetSpecification(
      class_names=class_names,
      # The search corpus is filtered down to embeddings from the targeted
      # dataset and having some foreground or background annotation of a class
      # in `class_names`.
      search_corpus_global_mask_fn=(
          lambda df: has_dataset_name(df) & has_some_annotation(df)),
      # For Xeno-Canto, we make sure that target species' background
      # vocalizations are not present in the 'xc_fg' corpus, and vice versa.
      search_corpus_classwise_mask_fn={
          'xc_fg':
              lambda df, class_name: ~df['bg_labels'].str.contains(class_name),
          'xc_bg':
              lambda df, class_name: ~df['label'].str.contains(class_name),
          'birdclef':
              lambda df, _: df['label'].map(lambda s: True),
      }[corpus_type],
      # We always draw from foreground-vocalizing species in Xeno-Canto to form
      # the collection of class representatives.
      class_representative_global_mask_fn=(
          lambda df: df['dataset_name'] == 'xc_downstream'),
      class_representative_classwise_mask_fn=(
          lambda df, class_name: df['label'].str.contains(class_name)),
      num_representatives_per_class=num_representatives_per_class,
  )


def get_config() -> config_dict.ConfigDict:
  """Creates a configuration dictionary for the MVP evaluation protocol.

  The MVP protocol evaluates on artificially rare Sapsucker Woods (SSW) species
  as well as on held-out Colombia and Hawaii species.

  Returns:
    The configuration dictionary for the MVP evaluation protocol.
  """
  config = config_dict.ConfigDict()
  config.tfds_data_dir = config_dict.FieldReference(_TFDS_DATA_DIR)
  # The model_callback is expected to be a Callable[[np.ndarray], np.ndarray].
  config.model_callback = lambda x: [0.0]
  # The PRNG seed controls the random subsampling of class representatives down
  # to the right number of when forming eval sets.
  config.rng_seed = 1234

  # TODO(bringingjoy): extend create_species_query to support returning multiple
  # queries for a given eval species.
  config.create_species_query = eval_lib.create_averaged_query
  config.score_search = eval_lib.cosine_similarity

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
    dataset_config.tfds_data_dir = config.tfds_data_dir

    ops = [
        _c('pipeline.OnlyKeep',
           names=['audio', 'label', 'bg_labels', 'recording_id', 'segment_id']),
        _c('pipeline.NormalizeAudio', target_gain=target_gain),
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
    # SSW species are artificially-rare, which means that we learned a species
    # representation for them during upstream training. Rather than using a
    # collection of species representatives, we use that learned representation
    # to perform search, and we correspondingly set
    # `num_representatives_per_class` to zero.
    if location == 'ssw':
      config.eval_set_specifications[f'artificially_rare_{corpus_type}'] = (
          _make_eval_set_config(
              location=location,
              corpus_type=corpus_type,
              num_representatives_per_class=0))
    # For downstream species, we sweep over {1, 2, 4, 8, 16} representatives
    # per class, and in each case we resample the collection of class
    # representatives 5 times to get confidence intervals on the metrics.
    else:
      for k, seed in itertools.product((1, 2, 4, 8, 16), range(1, 6)):
        config.eval_set_specifications[
            f'{location}_{corpus_type}_{k}_seed{seed}'] = (
                _make_eval_set_config(
                    location=location,
                    corpus_type=corpus_type,
                    num_representatives_per_class=k))

  return config
