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

"""Bird taxonomy dataset."""

import dataclasses
import functools
import resource
import tempfile
from typing import Any, Callable
import warnings

from chirp import audio_utils
from chirp.data import filter_scrub_utils as fsu
from chirp.data import tfds_features
from chirp.data.bird_taxonomy import premade_queries
from chirp.taxonomy import namespace_db
from etils import epath
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Bird taxonomy dataset of Xeno-Canto audio files.
"""

_CITATION = """
@inproceedings{vellinga2015xeno,
  title={The Xeno-canto Collection and its Relation to Sound Recognition and Classification.},
  author={Vellinga, Willem-Pier and Planqu{\'e}, Robert},
  booktitle={CLEF (Working Notes)},
  year={2015}
}

Credit for individual audio recordings can be viewed by visiting
https://xeno-canto.org/{xeno_canto_id}, and a given example's Xeno-Canto ID can
be retrieved from the 'filename' feature: 'XC{xeno_canto_id}.mp3'.
"""

# The maximum audio sequence length to consider if a localization function is
# provided. This is 5 * 60 seconds = 5 minutes.
_MAX_LOCALIZATION_LENGTH_S = 5 * 60

LocalizationFn = Callable[[Any, int, float], jnp.ndarray]


@dataclasses.dataclass
class BirdTaxonomyConfig(tfds.core.BuilderConfig):
  """The config used to generate multiple versions of BirdTaxonomy.

  Special note on processing queries: Because some queries don't make sense
  applying to the metadata dataframe, e.g. scrubbing, we make a disctinction
  between `data_processing_query` applied to the recordings' dataframe, and
  `metadata_processing_query` applied to the metadata (used in _info()).
  Checks are made downstream to ensure both dataframes encode consistent
  label spaces.
  """

  sample_rate_hz: int = 32_000
  resampling_method: str = 'polyphase'
  localization_fn: LocalizationFn | None = None
  interval_length_s: float | None = None
  data_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  metadata_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  class_list_name: str = 'xenocanto'


class BirdTaxonomy(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the bird taxonomy dataset."""

  VERSION = tfds.core.Version('2.1.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': (
          'Switched to higher sampling rate, added recording metadata '
          'features, switched to log-scaling in slice_peaked_audio.'
      ),
      '1.1.1': 'Added slice_peaked_tiny config.',
      '1.1.2': (
          'Kept previous tiny_config as reference, but also added a tiny'
          'version generated with queries.'
      ),
      '1.2.0': 'Added upstream data config.',
      '1.2.1': (
          'Added downstream data config. Fixed the upstream query.'
          'Bumped the taxonomy_info to 2022-07-18.'
      ),
      '1.2.2': (
          'Replacing any non-relevant foreground annotation in the'
          'downstream data with "ignore" class: downstream data only'
          'contains relevant annotations + "ignore" class.'
      ),
      '1.2.3': (
          'Removing any non-relevant annotation from foreground or '
          'background in downstream data: downstream data only'
          'contains relevant annotations. Also removing order, family and'
          'genus metadata, as those will be added in the TF-based'
          'processing pipeline.'
      ),
      '1.2.4': 'Adds a unique recording ID and a segment ID to all samples.',
      '1.2.5': 'Refactor Int16AsFloatTensor out of BirdTaxonomy.',
      '1.3.0': (
          'Added "upstream_full_length", "downstream_full_length", '
          '"upstream_ar_only_slice_peaked", and '
          '"upstream_ar_only_full_length" variants. Removed '
          '"slice_peaked_tiny_reference" variant.'
      ),
      '1.4.0': 'Added a seabird_sliced_peaked dataset.',
      '1.5.0': 'Updated ebird2021 taxonomy.',
      '2.0.0': (
          "Updated the upstream split to align with Coffee Farms and Hawai'i."
      ),
      '2.1.0': (
          "Added a 'class_representatives_slice_peaked' variant which contains "
          'recordings for High Sierras, Sierra Nevada, and Peru species in '
          'addition to recordings for artificially-rare and downstream species.'
      ),
  }
  BUILDER_CONFIGS = [
      # pylint: disable=unexpected-keyword-arg
      BirdTaxonomyConfig(
          name='slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          description=(
              'Chunked audio sequences processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      BirdTaxonomyConfig(
          name='slice_peaked_tiny',
          localization_fn=functools.partial(
              audio_utils.slice_peaked_audio, max_intervals=1
          ),
          interval_length_s=6.0,
          description=(
              'A tiny version of the slice_peaked dataset '
              'containing only two species'
          ),
          data_processing_query=fsu.QuerySequence([
              fsu.filter_in_class_list('species_code', 'tiny_species'),
              fsu.scrub_all_but_class_list('bg_species_codes', 'tiny_species'),
          ]),
          metadata_processing_query=fsu.QuerySequence([
              fsu.filter_in_class_list('species_code', 'tiny_species'),
          ]),
      ),
      BirdTaxonomyConfig(
          name='upstream_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=premade_queries.get_upstream_data_query(),
          metadata_processing_query=premade_queries.get_upstream_metadata_query(),
          description=(
              'Upstream data version with chunked audio sequences '
              'processed with chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      BirdTaxonomyConfig(
          name='upstream_ar_only_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=premade_queries.get_upstream_data_query(
              ar_only=True
          ),
          metadata_processing_query=premade_queries.get_upstream_metadata_query(),
          description=(
              'Upstream data version (AR-only) with chunked audio '
              'sequences processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      BirdTaxonomyConfig(
          name='downstream_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=premade_queries.get_downstream_data_query(),
          metadata_processing_query=premade_queries.get_downstream_metadata_query(),
          description=(
              'Downstream data version with chunked audio sequences '
              'processed with chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      BirdTaxonomyConfig(
          name='class_representatives_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=(
              premade_queries.get_class_representatives_data_query()
          ),
          metadata_processing_query=(
              premade_queries.get_class_representatives_metadata_query()
          ),
          description=(
              'All recordings available to be used as class representatives '
              '(namely recording for artificially-rare, downstream, High '
              'Sierras, Sierra Nevada, and Peru), processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
      ),
      BirdTaxonomyConfig(
          name='full_length',
          localization_fn=None,
          description='Full-length audio sequences.',
      ),
      BirdTaxonomyConfig(
          name='upstream_full_length',
          localization_fn=None,
          data_processing_query=premade_queries.get_upstream_data_query(),
          metadata_processing_query=premade_queries.get_upstream_metadata_query(),
          description='Upstream data with full-length audio sequences.',
      ),
      BirdTaxonomyConfig(
          name='upstream_ar_only_full_length',
          localization_fn=None,
          data_processing_query=premade_queries.get_upstream_data_query(
              ar_only=True
          ),
          metadata_processing_query=premade_queries.get_upstream_metadata_query(),
          description=(
              'Upstream data (AR-only) with full-length audio sequences.'
          ),
      ),
      BirdTaxonomyConfig(
          name='downstream_full_length',
          localization_fn=None,
          data_processing_query=premade_queries.get_downstream_data_query(),
          metadata_processing_query=premade_queries.get_downstream_metadata_query(),
          description='Downstream data with full-length audio sequences.',
      ),
      BirdTaxonomyConfig(
          name='seabird_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          description=(
              'Seabird dataset consisting of data '
              'with chunked audio sequences processed with '
              'chirp.audio_utils.slice_peaked_audio.'
          ),
          data_processing_query=fsu.QuerySequence([
              fsu.filter_in_class_list(
                  'species_code', 'ebird2021_global_seabirds'
              ),
              fsu.scrub_all_but_class_list(
                  'bg_species_codes', 'ebird2021_global_seabirds'
              ),
          ]),
          metadata_processing_query=fsu.QuerySequence([
              fsu.filter_in_class_list(
                  'species_code', 'ebird2021_global_seabirds'
              ),
          ]),
      ),
  ]

  GCS_URL = epath.Path('gs://chirp-public-bucket/xeno-canto')
  TAXONOMY_INFO_FILENAME = 'taxonomy_info_2022-07-18.json'

  def _load_taxonomy_metadata(self, disable_filtering: bool = False):
    """Loads the taxonomy for the dataset."""
    db = namespace_db.load_db()
    dataset_classes = list(
        db.class_lists[self.builder_config.class_list_name].classes
    )
    taxonomy_df = pd.DataFrame(dataset_classes, columns=['species_code'])
    if not disable_filtering:
      # We apply all the metadata processing queries
      taxonomy_df = fsu.apply_sequence(
          taxonomy_df, self.builder_config.metadata_processing_query
      )
    return taxonomy_df

  def _info(self) -> tfds.core.DatasetInfo:
    full_length = self.builder_config.localization_fn is None
    audio_feature_shape = [
        None
        if full_length
        else int(
            self.builder_config.sample_rate_hz
            * self.builder_config.interval_length_s
        )
    ]
    if tf.io.gfile.exists(self._data_dir):
      # If this data exists on disk, load the labels from there
      class_names = None
    else:
      # Load the class list relevant to the file
      class_names = self._load_taxonomy_metadata()['species_code'].tolist()

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio': tfds_features.Int16AsFloatTensor(
                shape=audio_feature_shape,
                sample_rate=self.builder_config.sample_rate_hz,
                encoding=tfds.features.Encoding.ZLIB,
            ),
            'recording_id': tfds.features.Scalar(dtype=tf.uint64),
            'segment_id': tfds.features.Scalar(dtype=tf.int64),
            'segment_start': tfds.features.Scalar(dtype=tf.uint64),
            'segment_end': tfds.features.Scalar(dtype=tf.uint64),
            'label': tfds.features.Sequence(
                tfds.features.ClassLabel(names=class_names)
            ),
            'bg_labels': tfds.features.Sequence(
                tfds.features.ClassLabel(names=class_names)
            ),
            'filename': tfds.features.Text(),
            'quality_score': tfds.features.Text(),
            'license': tfds.features.Text(),
            'altitude': tfds.features.Text(),
            'length': tfds.features.Text(),
            'bird_seen': tfds.features.Text(),
            'country': tfds.features.Text(),
            'latitude': tfds.features.Text(),
            'longitude': tfds.features.Text(),
            'playback_used': tfds.features.Text(),
            'recordist': tfds.features.Text(),
            'remarks': tfds.features.Text(),
            'sound_type': tfds.features.Text(),
        }),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/perch',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # Increase the file handle resource soft limit to the hard limit. The
    # dataset is large enough that it causes TFDS to hit the soft limit.
    _low, _high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_high, _high))

    # No checksum is found for the new taxonomy_info. dl_manager may raise
    # an error when removing the line below.
    dl_manager._force_checksums_validation = (
        False  # pylint: disable=protected-access
    )
    paths = dl_manager.download_and_extract({
        'taxonomy_info': (
            self.GCS_URL / self.TAXONOMY_INFO_FILENAME
        ).as_posix(),
    })
    # Load taxonomy_info, which is a superset of taxonomy_metadata that also
    # includes information on the Xeno-Canto files associated with each
    # species.
    taxonomy_info = pd.read_json(paths['taxonomy_info'])

    # Workaround for pandas<1.3.0's lack of multi-column explode. We set the
    # index to the non-exploding columns before applying pd.Series.explode
    # to the other columns and resetting the index.
    source_info = (
        taxonomy_info[
            taxonomy_info['xeno_canto_ids'].map(
                lambda xc_ids: bool(len(xc_ids))
            )
        ]
        .set_index([
            'species_code',
            'xeno_canto_query',
            'scientific_name',
            'species',
            'genus',
            'family',
            'order',
            'common_name',
        ])
        .apply(pd.Series.explode, axis=0)
        .reset_index()
    )
    # Rename columns to reflect the fact that they contain one value per row.
    renames = {
        'xeno_canto_ids': 'xeno_canto_id',
        'altitudes': 'altitude',
        'lengths': 'length',
        'countries': 'country',
        'file_formats': 'file_format',
        'latitudes': 'latitude',
        'licenses': 'license',
        'longitudes': 'longitude',
        'quality_scores': 'quality_score',
        'recordists': 'recordist',
        'sound_types': 'sound_type',
    }
    source_info = source_info.rename(renames, axis=1)

    get_format = lambda s: s['file_format']
    get_xc_id = lambda s: s['xeno_canto_id']
    to_name = lambda s: f"{s['species_code']}/XC{get_xc_id(s)}.{get_format(s)}"
    source_info['url'] = source_info.apply(
        lambda s: self.GCS_URL / f'audio-data/{to_name(s)}', axis=1
    )

    # Apply all the processing queries.
    source_info = fsu.apply_sequence(
        source_info, self.builder_config.data_processing_query
    )

    # Remap '' and 'no score' scores to 'E' (the worst score).
    source_info['quality_score'] = source_info['quality_score'].map(
        lambda s: 'E' if s in ('', 'no score') else s
    )

    # Remap None to '' for the 'latitude' and 'longitude' columns.
    for column in ['latitude', 'longitude']:
      source_info[column] = source_info[column].map(lambda s: s or '')

    return {
        'train': self._generate_examples(source_info=source_info),
    }

  def _generate_examples(self, source_info: pd.DataFrame):
    beam = tfds.core.lazy_imports.apache_beam
    librosa = tfds.core.lazy_imports.librosa

    def _process_example(row):
      recording_id, source = row
      with tempfile.NamedTemporaryFile(
          mode='w+b', suffix=source['url'].suffix
      ) as f:
        f.write(source['url'].read_bytes())
        # librosa outputs lots of warnings which we can safely ignore when
        # processing all Xeno-Canto files and PySoundFile is unavailable.
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          audio, _ = librosa.load(
              f.name,
              sr=self.builder_config.sample_rate_hz,
              res_type=self.builder_config.resampling_method,
          )
          # Resampling can introduce artifacts that push the signal outside the
          # [-1, 1) interval.
          audio = np.clip(audio, -1.0, 1.0 - (1.0 / float(1 << 15)))
      # Skip empty audio files.
      if audio.shape[0] == 0 or np.max(np.abs(audio)) == 0.0:
        return None
      # The scrubbed foreground annotations are replaced by ''. When this is the
      # case, we translate this annotation into []  rather than [''].
      foreground_label = (
          [source['species_code']] if source['species_code'] else []
      )
      return source['xeno_canto_id'], {
          'audio': audio,
          'recording_id': recording_id,
          'segment_id': -1,
          'segment_start': 0,
          'segment_end': len(audio),
          'label': foreground_label,
          'bg_labels': source['bg_species_codes'],
          'filename': source['url'].name,
          'quality_score': source['quality_score'],
          'license': source['license'],
          'altitude': source['altitude'],
          'length': source['length'],
          'bird_seen': source['bird_seen'],
          'country': source['country'],
          'latitude': source['latitude'],
          'longitude': source['longitude'],
          'playback_used': source['playback_used'],
          'recordist': source['recordist'],
          'remarks': source['remarks'],
          'sound_type': source['sound_type'],
      }

    if self.builder_config.localization_fn:

      def localize_intervals_fn(args):
        key, example = args
        sample_rate_hz = self.builder_config.sample_rate_hz
        interval_length_s = self.builder_config.interval_length_s
        target_length = int(sample_rate_hz * interval_length_s)

        audio = example['audio']

        # We limit audio sequence length to _MAX_LOCALIZATION_LENGTH_S when
        # localizing intervals because the localization function can result in
        # very large memory consumption for long audio sequences.
        max_length = sample_rate_hz * _MAX_LOCALIZATION_LENGTH_S
        if audio.shape[0] > max_length:
          audio = audio[:max_length]

        audio = audio_utils.pad_to_length_if_shorter(audio, target_length)
        # Pass padded audio to avoid localization_fn having to pad again
        audio_intervals = self.builder_config.localization_fn(
            audio, sample_rate_hz, interval_length_s
        ).tolist()

        if not audio_intervals:
          # If no peaks were found, we take the first segment of the
          # recording to avoid discarding it entirely
          audio_intervals = [(0, target_length)]
        interval_examples = []
        for i, (start, end) in enumerate(audio_intervals):
          interval_examples.append((
              f'{key}_{i}',
              {
                  **example,
                  'audio': audio[start:end],
                  'segment_id': i,
                  'segment_start': start,
                  'segment_end': end,
              },
          ))
        return interval_examples

    else:
      localize_intervals_fn = None

    for i, key_and_example in enumerate(
        map(_process_example, source_info.iterrows())
    ):
      # Since the audio files have variable length, the JAX compilation cache
      # can use up a large amount of memory after a while.
      if i % 100 == 0:
        jax.clear_caches()

      # Skip empty audio files.
      if key_and_example is None:
        continue

      if localize_intervals_fn:
        for key_and_example in localize_intervals_fn(key_and_example):
          yield key_and_example
      else:
        yield key_and_example
