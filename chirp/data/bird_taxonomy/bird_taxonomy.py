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

"""Bird taxonomy dataset."""

import dataclasses
import functools
import json
import os
import tempfile
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import warnings

from absl import logging
from chirp import audio_utils
from chirp.data import filter_scrub_utils as fsu
from chirp.data.bird_taxonomy import premade_queries
from etils import epath
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

LocalizationFn = Callable[[Any, int, float], Sequence[Tuple[int, int]]]


@dataclasses.dataclass
class BirdTaxonomyConfig(tfds.core.BuilderConfig):
  """The config used to generate multiple versions of BirdTaxonomy.

  Special note on processing queries: Because some queries don't make sense
  applying to the metadata dataframe, e.g. scrubbing, we make a disctinction
  between `data_processing_query` applied to the recordings' dataframe, and
  `metadata_processing_query` applied to the metadata (used in _info()).
  Checks are made downstream to ensure both dataframes encode consistent
  label spaces.
  Second note, we kept the original implementation of the tiny dataset, renamed
  here as `reference`, to ensure consistency with the new implementation based
  on queries.
  """
  sample_rate_hz: int = 32_000
  resampling_method: str = 'polyphase'
  localization_fn: Optional[LocalizationFn] = None
  interval_length_s: Optional[float] = None
  data_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  metadata_processing_query: fsu.QuerySequence = fsu.QuerySequence(queries=[])
  tiny_reference: bool = False


class Int16AsFloatTensor(tfds.features.Audio):
  """An int16 tfds.features.Tensor represented as a float32 in [-1, 1).

  Examples are stored as int16 tensors but encoded from and decoded into float32
  tensors in the [-1, 1) range (1 is excluded because we divide the
  [-2**15, 2**15 - 1] interval by 2**15).
  """
  INT16_SCALE = float(1 << 15)

  def __init__(
      self,
      *,
      file_format: Optional[str] = None,
      shape: tfds.typing.Shape,
      dtype: tf.dtypes.DType = tf.float32,
      sample_rate: tfds.typing.Dim,
      encoding: Union[str,
                      tfds.features.Encoding] = tfds.features.Encoding.NONE,
      doc: tfds.features.DocArg = None):
    del file_format
    del dtype

    self._int16_tensor_feature = tfds.features.Tensor(
        shape=shape, dtype=tf.int16, encoding=encoding)

    super().__init__(
        file_format=None,
        shape=shape,
        dtype=tf.float32,
        sample_rate=sample_rate,
        encoding=encoding,
        doc=doc)

  def get_serialized_info(self):
    return self._int16_tensor_feature.get_serialized_info()

  def encode_example(self, example_data):
    if not isinstance(example_data, np.ndarray):
      example_data = np.array(example_data, dtype=np.float32)
    if example_data.dtype != np.float32:
      raise ValueError('dtype should be float32')
    if (example_data.min() < -1.0 or
        example_data.max() > 1.0 - (1.0 / self.INT16_SCALE)):
      raise ValueError('values should be in [-1, 1)')
    return self._int16_tensor_feature.encode_example(
        (example_data * self.INT16_SCALE).astype(np.int16))

  def decode_example(self, tfexample_data):
    int16_scale = tf.constant(self.INT16_SCALE, dtype=tf.float32)
    decoded_data = tf.cast(
        self._int16_tensor_feature.decode_example(tfexample_data), tf.float32)
    return decoded_data / int16_scale


class BirdTaxonomy(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for the bird taxonomy dataset."""

  VERSION = tfds.core.Version('1.2.2')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': ('Switched to higher sampling rate, added recording metadata '
                'features, switched to log-scaling in slice_peaked_audio.'),
      '1.1.1': 'Added slice_peaked_tiny config.',
      '1.1.2': 'Kept previous tiny_config as reference, but also added a tiny'
               'version generated with queries.',
      '1.2.0': 'Added upstream data config.',
      '1.2.1': 'Added downstream data config. Fixed the upstream query.'
               'Bumped the taxonomy_info to 2022-07-18.',
      '1.2.2': 'Replacing any non-relevant foreground annotation in the'
               'downstream data with "ignore" class: downstream data only'
               'contains relevant annotations + "ignore" class.',
  }
  TINY_SPECIES = ('ostric2', 'piebar1')
  BUILDER_CONFIGS = [
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          description=('Chunked audio sequences processed with '
                       'chirp.audio_utils.slice_peaked_audio.')),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='slice_peaked_tiny_reference',
          localization_fn=functools.partial(
              audio_utils.slice_peaked_audio, max_intervals=1),
          interval_length_s=6.0,
          tiny_reference=True,
          description=('A reference tiny version of the slice_peaked dataset '
                       'containing only two species, built with using pandas'
                       'built-in functions.')),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='slice_peaked_tiny',
          localization_fn=functools.partial(
              audio_utils.slice_peaked_audio, max_intervals=1),
          interval_length_s=6.0,
          description=('A tiny version of the slice_peaked dataset '
                       'containing only two species, built using homemade '
                       'queries.'),
          data_processing_query=fsu.QuerySequence([
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      'mask_op': fsu.MaskOp.IN,
                      'op_kwargs': {
                          'key': 'species_code',
                          'values': list(TINY_SPECIES)
                      }
                  }),
              fsu.Query(
                  op=fsu.TransformOp.SCRUB_ALL_BUT,
                  kwargs={
                      'key': 'bg_species_codes',
                      'values': list(TINY_SPECIES)
                  })
          ]),
          metadata_processing_query=fsu.QuerySequence([
              fsu.Query(
                  op=fsu.TransformOp.FILTER,
                  kwargs={
                      'mask_op': fsu.MaskOp.IN,
                      'op_kwargs': {
                          'key': 'species_code',
                          'values': list(TINY_SPECIES)
                      }
                  }),
          ])),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='upstream_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=premade_queries.get_upstream_data_query(),
          metadata_processing_query=premade_queries.get_upstream_metadata_query(
          ),
          description=('Upstream data version with chunked audio sequences '
                       'processed with chirp.audio_utils.slice_peaked_audio.')),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='downstream_slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          data_processing_query=premade_queries.get_downstream_data_query(),
          metadata_processing_query=premade_queries
          .get_downstream_metadata_query(),
          description=('Downstream data version with chunked audio sequences '
                       'processed with chirp.audio_utils.slice_peaked_audio.')),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='full_length',
          localization_fn=None,
          description='Full-length audio sequences.'),
  ]

  GCS_URL = epath.Path('gs://chirp-public-bucket/xeno-canto')
  TAXONOMY_INFO_FILENAME = 'taxonomy_info_2022-07-18.json'

  def _load_taxonomy_metadata(self, disable_filtering=False) -> pd.DataFrame:
    file_path = (
        epath.Path(__file__).parent / f'metadata/taxonomy_metadata.json')
    # The taxonomy_metadata.json file contains a taxonomy tree organized as
    # Dict[str, Dict[str, Dict[str, Sequence[str]]]] which maps order name
    # to family name to genus name to a list of species codes.
    order_to_families = json.loads(file_path.read_text())
    rows = []
    for order, family_to_genera in order_to_families.items():
      for family, genus_to_species_codes in family_to_genera.items():
        for genus, species_codes in genus_to_species_codes.items():
          for species_code in species_codes:
            # When building tiny reference, we ensure upfront that only
            # TINY_SPECIES are added. Instead, the `query-based` tiny version
            # first adds everything, and filters out afterwards.
            if disable_filtering or not self.builder_config.tiny_reference or species_code in self.TINY_SPECIES:
              rows.append({
                  'species_code': species_code,
                  'genus': genus,
                  'family': family,
                  'order': order
              })
    df = pd.DataFrame(rows)
    # At that point, the dataframe contains all possible species. Now we apply
    # filtering operations.
    if not disable_filtering and not self.builder_config.tiny_reference:
      # We apply all the metadata processing queries
      df = fsu.apply_sequence(df, self.builder_config.metadata_processing_query)
    return df

  def _info(self) -> tfds.core.DatasetInfo:
    # The taxonomy_metadata dataframe is a lightweight subset of the
    # TAXONOMY_INFO_FILENAME dataframe stored on GCS. More specifically, it
    # drops all columns that are not needed to construct the 'label', 'genus',
    # 'family', and 'order' class sets. This lets us avoid downloading any data
    # outside of the _split_generators function.
    taxonomy_metadata = self._load_taxonomy_metadata()
    class_names = {
        level: sorted(set(taxonomy_metadata[level]))
        for level in ('species_code', 'genus', 'family', 'order')
    }

    full_length = self.builder_config.localization_fn is None
    audio_feature_shape = [
        None if full_length else int(self.builder_config.sample_rate_hz *
                                     self.builder_config.interval_length_s)
    ]

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio':
                Int16AsFloatTensor(
                    shape=audio_feature_shape,
                    sample_rate=self.builder_config.sample_rate_hz,
                    encoding=tfds.features.Encoding.ZLIB,
                ),
            'segment_start':
                tfds.features.Scalar(dtype=tf.uint64),
            'segment_end':
                tfds.features.Scalar(dtype=tf.uint64),
            'label':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=class_names['species_code'])
                ),
            'bg_labels':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=class_names['species_code'])
                ),
            'genus':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=class_names['genus'])),
            'family':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=class_names['family'])),
            'order':
                tfds.features.Sequence(
                    tfds.features.ClassLabel(names=class_names['order'])),
            'filename':
                tfds.features.Text(),
            'quality_score':
                tfds.features.Text(),
            'license':
                tfds.features.Text(),
            'altitude':
                tfds.features.Text(),
            'length':
                tfds.features.Text(),
            'bird_seen':
                tfds.features.Text(),
            'country':
                tfds.features.Text(),
            'latitude':
                tfds.features.Text(),
            'longitude':
                tfds.features.Text(),
            'playback_used':
                tfds.features.Text(),
            'recordist':
                tfds.features.Text(),
            'remarks':
                tfds.features.Text(),
            'sound_type':
                tfds.features.Text(),
        }),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/chirp',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # No checksum is found for the new taxonomy_info. dl_manager may raise
    # an error when removing the line below.
    dl_manager._force_checksums_validation = False
    paths = dl_manager.download_and_extract({
        'taxonomy_info':
            (self.GCS_URL / self.TAXONOMY_INFO_FILENAME).as_posix(),
    })
    # Load taxonomy_info, which is a superset of taxonomy_metadata that also
    # includes information on the Xeno-Canto files associated with each
    # species.
    taxonomy_info = pd.read_json(paths['taxonomy_info'])
    if not taxonomy_info[[
        'species_code', 'genus', 'family', 'order'
    ]].sort_values(
        by='species_code', axis=0, ignore_index=True).equals(
            self._load_taxonomy_metadata(disable_filtering=True).sort_values(
                by='species_code', axis=0, ignore_index=True)):
      raise RuntimeError('Downloaded taxonomy_info dataframe is incompatible '
                         'with the taxonomy_metadata dataframe.')

    # Workaround for pandas<1.3.0's lack of multi-column explode. We set the
    # index to the non-exploding columns before applying pd.Series.explode
    # to the other columns and resetting the index.
    source_info = taxonomy_info[taxonomy_info['xeno_canto_ids'].map(
        lambda xc_ids: bool(len(xc_ids)))].set_index([
            'species_code',
            'xeno_canto_query',
            'scientific_name',
            'species',
            'genus',
            'family',
            'order',
            'common_name',
        ]).apply(
            pd.Series.explode, axis=0).reset_index()
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
        lambda s: self.GCS_URL / f'audio-data/{to_name(s)}', axis=1)

    # To generate the reference tiny version, we filter/scrub using built-in
    # dataframe functions only, not queries.
    if self.builder_config.tiny_reference:
      source_info = source_info[source_info['species_code'].isin(
          self.TINY_SPECIES)]
      source_info['bg_species_codes'] = source_info['bg_species_codes'].map(
          lambda codes: [c for c in codes if c in self.TINY_SPECIES])
    else:
      # We apply all the processing queries.
      source_info = fsu.apply_sequence(
          source_info, self.builder_config.data_processing_query)
    # Remap '' and 'no score' scores to 'E' (the worst score).
    source_info['quality_score'] = source_info['quality_score'].map(
        lambda s: 'E' if s in ('', 'no score') else s)

    # Remap None to '' for the 'latitude' and 'longitude' columns.
    for column in ['latitude', 'longitude']:
      source_info[column] = source_info[column].map(lambda s: s or '')

    # Propagate "ignore" label to genus, family and order metadata.
    for column in ['genus', 'family', 'order']:
      source_info[column] = source_info.apply(
          lambda rec: 'ignore'
          if rec['species_code'] == 'ignore' else rec[column],
          axis=1)

    return {
        'train': self._generate_examples(source_info=source_info),
    }

  def _generate_examples(self, source_info: pd.DataFrame):
    beam = tfds.core.lazy_imports.apache_beam
    librosa = tfds.core.lazy_imports.librosa

    def _process_example(source):
      with tempfile.NamedTemporaryFile(
          mode='w+b', suffix=source['url'].suffix) as f:
        f.write(source['url'].read_bytes())
        # librosa outputs lots of warnings which we can safely ignore when
        # processing all Xeno-Canto files and PySoundFile is unavailable.
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          audio, _ = librosa.load(
              f.name,
              sr=self.builder_config.sample_rate_hz,
              res_type=self.builder_config.resampling_method)
          # Resampling can introduce artifacts that push the signal outside the
          # [-1, 1) interval.
          audio = np.clip(audio, -1.0, 1.0 - (1.0 / float(1 << 15)))

      return source['xeno_canto_id'], {
          'audio': audio,
          'segment_start': 0,
          'segment_end': len(audio),
          'label': [source['species_code']],
          'bg_labels': source['bg_species_codes'],
          'genus': [source['genus']],
          'family': [source['family']],
          'order': [source['order']],
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

    pipeline = (
        beam.Create(source for _, source in source_info.iterrows())
        | beam.Map(_process_example))

    if self.builder_config.localization_fn:

      def _localize_intervals(args):
        key, example = args
        sample_rate_hz = self.builder_config.sample_rate_hz
        interval_length_s = self.builder_config.interval_length_s
        target_length = int(sample_rate_hz * interval_length_s)

        audio = audio_utils.pad_to_length_if_shorter(example['audio'],
                                                     target_length)
        # Pass padded audio to avoid localization_fn having to pad again
        audio_intervals = self.builder_config.localization_fn(
            audio, sample_rate_hz, interval_length_s).tolist()

        if not audio_intervals:
          # If no peaks were found, we take the first segment of the
          # recording to avoid discarding it entirely
          audio_intervals = [(0, target_length)]
        interval_examples = []
        for i, (start, end) in enumerate(audio_intervals):
          interval_examples.append((f'{key}_{i}', {
              **example, 'audio': audio[start:end],
              'segment_start': start,
              'segment_end': end
          }))
        return interval_examples

      pipeline = pipeline | beam.FlatMap(_localize_intervals)

    return pipeline
