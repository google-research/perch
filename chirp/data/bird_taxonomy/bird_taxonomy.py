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
import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

from absl import logging
from chirp import audio_utils
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

KeyExample = Tuple[Union[str, int], Dict[str, Any]]
LocalizationFn = Callable[[KeyExample, int, float], Sequence[KeyExample]]


@dataclasses.dataclass
class BirdTaxonomyConfig(tfds.core.BuilderConfig):
  sample_rate_hz: int = 22_500
  resampling_method: str = 'polyphase'
  localization_fn: Optional[LocalizationFn] = None
  interval_length_s: Optional[float] = None


class Int16AsFloatTensor(tfds.features.Tensor):
  """An int16 tfds.features.Tensor represented as a float32 in [-1, 1).

  Examples are stored as int16 tensors but encoded from and decoded into float32
  tensors in the [-1, 1) range (1 is excluded because we divide the
  [-2**15, 2**15 - 1] interval by 2**15).
  """
  INT16_SCALE = float(1 << 15)

  def __init__(self,
               *,
               shape: tfds.typing.Shape,
               sample_rate: int,
               encoding: tfds.features.Encoding = tfds.features.Encoding.NONE):
    self._int16_tensor_feature = tfds.features.Tensor(
        shape=shape, dtype=tf.int16, encoding=encoding)
    # Note: We use sample_rate instead of sample_rate_hz to match
    # tfds.features.Audio.
    self.sample_rate = sample_rate
    super().__init__(shape=shape, dtype=tf.float32, encoding=encoding)

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

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = [
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='slice_peaked',
          localization_fn=audio_utils.slice_peaked_audio,
          interval_length_s=6.0,
          description=('Chunked audio sequences processed with '
                       'chirp.audio_utils.slice_peaked_audio.')),
      BirdTaxonomyConfig(  # pylint: disable=unexpected-keyword-arg
          name='full_length',
          localization_fn=None,
          description='Full-length audio sequences.'),
  ]

  GCS_URL = epath.Path('gs://chirp-public-bucket/xeno-canto')
  TAXONOMY_INFO_FILENAME = 'taxonomy_info_2022-05-21.json'

  def _load_taxonomy_metadata(self) -> pd.DataFrame:
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
            rows.append({
                'species_code': species_code,
                'genus': genus,
                'family': family,
                'order': order
            })
    return pd.DataFrame(rows)

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
        }),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/chirp',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
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
            self._load_taxonomy_metadata().sort_values(
                by='species_code', axis=0, ignore_index=True)):
      raise RuntimeError('Downloaded taxonomy_info dataframe is incompatible '
                         'with the taxonomy_metadata dataframe.')

    # Workaround for pandas<1.3.0's lack of multi-column explode. We set the
    # index to the non-exploding columns before applying pd.Series.explode
    # to the other columns and resetting the index.
    source_info = taxonomy_info[taxonomy_info['xeno_canto_ids'].map(
        lambda xc_ids: bool(len(xc_ids)))].set_index([
            'species_code', 'xeno_canto_query', 'scientific_name', 'species',
            'genus', 'family', 'order', 'common_name'
        ]).apply(
            pd.Series.explode, axis=0).reset_index()
    # Rename columns to reflect the fact that they contain one value per row.
    renames = {
        'xeno_canto_ids': 'xeno_canto_id',
        'xeno_canto_formats': 'xeno_canto_format',
        'xeno_canto_quality_scores': 'xeno_canto_quality_score',
        'xeno_canto_licenses': 'xeno_canto_license',
    }
    source_info = source_info.rename(renames, axis=1)

    # Remap '' and 'no score' scores to 'E' (the worst score).
    clean_up_score = lambda s: 'E' if s in ('', 'no score') else s
    source_info['xeno_canto_quality_score'] = source_info[
        'xeno_canto_quality_score'].map(clean_up_score)

    get_format = lambda s: s['xeno_canto_format']
    get_xc_id = lambda s: s['xeno_canto_id']
    to_name = lambda s: f"{s['species_code']}/XC{get_xc_id(s)}.{get_format(s)}"
    source_info['url'] = source_info.apply(
        lambda s: self.GCS_URL / f'audio-data/{to_name(s)}', axis=1)

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
          'label': [source['species_code']],
          'bg_labels': source['xeno_canto_bg_species_codes'],
          'genus': [source['genus']],
          'family': [source['family']],
          'order': [source['order']],
          'filename': source['url'].name,
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

        audio = example['audio']
        audio_intervals = [
            audio_utils.pad_to_length_if_shorter(audio,
                                                 target_length)[:target_length]
        ]
        audio_intervals.extend(
            self.builder_config.localization_fn(example['audio'],
                                                sample_rate_hz,
                                                interval_length_s))
        common_features = [(k, v) for k, v in example.items() if k != 'audio']
        return [(f'{key}_{i}', dict([('audio', interval)] + common_features))
                for i, interval in enumerate(audio_intervals)]

      pipeline = pipeline | beam.FlatMap(_localize_intervals)

    return pipeline
