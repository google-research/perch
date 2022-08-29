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

"""Soundscape datasets."""

import collections
import dataclasses
import json
import tempfile
from typing import Callable, Dict, Optional, Set, Tuple, List, Any
import warnings

from absl import logging
from chirp import audio_utils
from chirp import path_utils
from chirp.data.bird_taxonomy import bird_taxonomy
from etils import epath
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

_DESCRIPTION = """
Soundscape datasets.
"""

_CITATION = """
@inproceedings{kahl2019overview,
  title={Overview of BirdCLEF 2019: large-scale bird recognition in soundscapes},
  author={Kahl, Stefan and St{\"o}ter, Fabian-Robert and Go{\"e}au, Herv{\'e} and Glotin, Herv{\'e} and Planque, Robert and Vellinga, Willem-Pier and Joly, Alexis},
  booktitle={Working Notes of CLEF 2019-Conference and Labs of the Evaluation Forum},
  number={2380},
  pages={1--9},
  year={2019},
  organization={CEUR}
}
"""

_AUDIO_EXTENSIONS = ['.flac', '.wav']

_DEPRECATED2NEW = {'reevir1': 'reevir'}


def load_birdclef_metadata(root: epath.Path) -> pd.DataFrame:
  """The `metadata_load_fn` for Birdclef2019-based configs."""
  metadata_path = root / 'birdclef2019' / 'metadata'
  df = []
  bar = tqdm.tqdm(metadata_path.iterdir())
  bar.set_description('Loading BirdClef2019 metadata.')
  for path in bar:
    with path.open('rb') as f:
      df.append(pd.read_json(f, typ='series'))
  df = pd.concat(df, axis=1).T
  df['fid'] = df['FileName']
  return df


@dataclasses.dataclass
class SoundscapesConfig(bird_taxonomy.BirdTaxonomyConfig):
  """Dataset configuration for Soundscape datasets.

  Attributes:
    audio_source: The name of the folder from where the audio will be fetched.
    metadata_fields: Because the metadata fields don't always appear with the
      same names, and because we don't necesarily care about every field, we
      specify the fields we're interested in keeping, as well how to map fields'
      names. Specifically, the keys are the names that will appear in the final
      tf.Example, while the values are the names of the fields of interest as
      they appear in the metadata files.
    metadata_load_fn: Because the metadata don't always appear under the same
      format, we specify for each config the way to load metadata. This function
      outputs a dataframe, where each row contains the metadata for some audio
      file.
    keep_unknown_annotation: An "unknown" annotations appears in some datasets.
      This boolean decides whether it should keep this annotation (and
      therefore) add a species named "unknown" in the label space, or just scrub
      all "unknown" annotations.
    restrict_label_space: If set to True, the label space will only include the
      species that are actually present in the current dataset, plus an eventual
      "unknown" class. If set to False, the label space will include the full
      Xeno-Canto taxonomy (~ 11k species).
  """
  audio_source: str = ''
  metadata_fields: Optional[Dict[str, str]] = None
  metadata_load_fn: Optional[Callable[[epath.Path], pd.DataFrame]] = None
  keep_unknown_annotation: bool = False
  restrict_label_space: bool = True
  supervised: bool = True
  audio_dir = epath.Path('gs://chirp-public-bucket/soundscapes')


class Soundscapes(bird_taxonomy.BirdTaxonomy):
  """DatasetBuilder for soundscapes data."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release. The label set corresponds to the full '
               'set of ~11 000 Xeno-Canto species.',
      '1.0.1': 'The label set is now restricted to the species present in each'
               'dataset.',
  }
  BUILDER_CONFIGS = [
      SoundscapesConfig(  # pylint: disable=unexpected-keyword-arg
          name='caples',  # TODO(mboudiaf) Try to interface caples metadata.
          audio_source='caples/audio',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          description=('Annotated Caples recordings from 2018/2019.')),
      SoundscapesConfig(  # pylint: disable=unexpected-keyword-arg
          name='birdclef2019_colombia',
          audio_source='birdclef2019/audio',
          metadata_load_fn=load_birdclef_metadata,
          interval_length_s=5.0,
          metadata_fields={
              'country': 'Country',
              'longitude': 'Longitude',
              'latitude': 'Latitude',
              'altitude': 'Elevation',
              'recordist': 'AuthorID'
          },
          localization_fn=audio_utils.slice_peaked_audio,
          description=(
              'Colombian recordings from the Birdclef 2019 challenge.')),
      SoundscapesConfig(  # pylint: disable=unexpected-keyword-arg
          name='birdclef2019_ssw',
          audio_source='birdclef2019/audio',
          metadata_load_fn=load_birdclef_metadata,
          interval_length_s=5.0,
          metadata_fields={
              'country': 'Country',
              'longitude': 'Longitude',
              'latitude': 'Latitude',
              'altitude': 'Elevation',
              'recordist': 'AuthorID'
          },
          localization_fn=audio_utils.slice_peaked_audio,
          description=('SSW recordings from the Birdclef 2019 challenge.')),
      SoundscapesConfig(  # pylint: disable=unexpected-keyword-arg
          name='high_sierras',
          audio_source='high_sierras/audio',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          description=('High Sierras recordings.')),
  ]

  def _load_segments(self, dl_manager):
    """Load the dataframe of all segments from metadata/."""

    paths = dl_manager.download_and_extract({
        'segments': (self.builder_config.audio_dir / 'metadata' /
                     f'{self.builder_config.name}.csv').as_posix(),
    })
    segment_path = paths['segments']
    segments = pd.read_csv(segment_path)
    segments['ebird_codes'] = segments['ebird_codes'].apply(
        lambda codes: codes.split())

    # Map deprecated ebird codes to new ones.
    segments['ebird_codes'] = segments['ebird_codes'].apply(lambda codes: [
        code if code not in _DEPRECATED2NEW else _DEPRECATED2NEW[code]
        for code in codes
    ])

    # Potentially remove all 'unknown' annotations.
    if not self.builder_config.keep_unknown_annotation:
      segments['ebird_codes'] = segments['ebird_codes'].apply(
          lambda codes: [code for code in codes if code != 'unknown'])

    # Keep only segments whose annotations haven't been emptied
    segments = segments[segments['ebird_codes'].apply(
        lambda codes: bool(len(codes)))]

    return segments

  def _load_taxonomy_metadata(self):
    """Loads the taxonomy containing currently present species."""

    # Load full taxonomy data
    taxonomy_metadata = super()._load_taxonomy_metadata()

    if self.builder_config.keep_unknown_annotation:
      # Add the 'unknown' spepcies
      taxonomy_metadata = taxonomy_metadata.append(
          {
              level: 'unknown'
              for level in ['species_code', 'genus', 'family', 'order']
          },
          ignore_index=True)

    if self.builder_config.restrict_label_space:
      # Load the set of species that are present in the current configuration
      present_species_path = path_utils.get_absolute_epath(
          'data/soundscapes/metadata/present_species.json')
      with open(present_species_path, 'r') as f:
        present_species = set(json.load(f)[self.builder_config.name])

      present_species = {
          _DEPRECATED2NEW[s] if s in _DEPRECATED2NEW else s
          for s in present_species
      }
      # Ensure that present species are part of the global taxonomy.
      all_species = set(taxonomy_metadata['species_code'].unique().tolist())
      if not present_species.issubset(all_species):
        logging.warning(
            'The following species do not belong to the global taxonomy %s',
            str(present_species.difference(all_species)))

      taxonomy_metadata = taxonomy_metadata[taxonomy_metadata[
          'species_code'].apply(lambda x: x in present_species)]
    return taxonomy_metadata

  def _info(self) -> tfds.core.DatasetInfo:

    taxonomy_metadata = self._load_taxonomy_metadata()
    class_names = {
        level: sorted(set(taxonomy_metadata[level]))
        for level in ('species_code', 'genus', 'family', 'order')
    }
    logging.info('Currently considering a total of %s species.',
                 len(class_names['species_code']))
    audio_feature_shape = (int(self.builder_config.interval_length_s *
                               self.builder_config.sample_rate_hz),)
    common_features = {
        'audio':
            bird_taxonomy.Int16AsFloatTensor(
                shape=audio_feature_shape,
                sample_rate=self.builder_config.sample_rate_hz,
                encoding=tfds.features.Encoding.ZLIB,
            ),
        'label':
            tfds.features.Sequence(
                tfds.features.ClassLabel(names=class_names['species_code'])),
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
        'segment_start':
            tfds.features.Scalar(dtype=tf.uint64),
        'segment_end':
            tfds.features.Scalar(dtype=tf.uint64),
    }
    if self.builder_config.metadata_load_fn is not None:
      if self.builder_config.metadata_fields is None:
        raise ValueError("If a 'metadata_load_fn' is specified, then the"
                         "'metadata_fields' mapping must also be specied.")
      additional_features = {
          k: tfds.features.Text()
          for k in self.builder_config.metadata_fields.keys()
      }
      common_features.update(additional_features)
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(common_features),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/chirp',
        citation=_CITATION,
    )

  def _combine_with_full_taxonomy(self, segments):
    """Combine the segment with genus, family and order information.

    Args:
      segments: The dataframe containing the segments, each of which is
        described by 4 fields: 'fid', 'start_time_s', 'end_time_s' and
        'ebird_codes'.

    Returns:
      segments: The dataframe containing the segmens, now described by the 4
      columns described in Args, plus 'genus', 'family' and 'order'.
    """
    species2taxonomy = {
        species: metadata for species, metadata in
        self._load_taxonomy_metadata().groupby('species_code')
    }
    for level in ['family', 'genus', 'order']:

      def species2level(species, level=level):
        # `species2taxonomy[s][level]` yields a pd.Series with a single
        # element containing the level (e.g. order) associated to this species.
        # To recover the string element, we take the first element.
        return [species2taxonomy[s][level].iloc[0] for s in species]

      segments[level] = segments['ebird_codes'].apply(species2level)
    return segments

  def _combine_with_metadata(self, segments):
    """Combine segments with whatever metadata is available for this dataset."""

    # Load the dataframe containing the metadata. Each row describes some audio
    # file, and the dataframe should contain the 'fid' column, which acts as the
    # key to match with segments.
    metadata_df = self.builder_config.metadata_load_fn(
        self.builder_config.audio_dir)
    fid_to_metadata_index = metadata_df.groupby('fid').groups
    combined_segments = []
    bar = tqdm.tqdm(segments.iterrows(), total=len(segments))
    bar.set_description('Combining segments will full metadata.')
    for _, segment in bar:
      fid = segment['fid']
      segment_metadata = metadata_df.loc[fid_to_metadata_index[fid]]
      if segment_metadata.empty:
        logging.warning('MediaId %d not found in metadata', fid)

      for compatible_key, metadata_key in self.builder_config.metadata_fields.items(
      ):
        segment[compatible_key] = str(segment_metadata[metadata_key].iloc[0])
      combined_segments.append(segment)
    concat_segments = pd.concat(combined_segments, axis=1).T
    return concat_segments

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):

    dl_manager._force_checksums_validation = False
    # Find all file present in the audio/directory
    audio_path = self.builder_config.audio_dir / self.builder_config.audio_source
    all_audio_filenames = list(audio_path.iterdir())

    if self.builder_config.supervised:
      # For supervised data, we first grab the annotated segments.
      segments = self._load_segments(dl_manager)

      # Add genus, family, order information for each segment.
      segments = self._combine_with_full_taxonomy(segments)

      # If specified, combine the segments with additional metadata
      # (e.g Country).
      if self.builder_config.metadata_load_fn is not None:
        segments = self._combine_with_metadata(segments)

      segments = segments.rename({
          'fid': 'filename',
          'ebird_codes': 'label'
      },
                                 axis=1)

      stem2path = {
          fpath.stem.split('.')[0]: fpath.parts[-1]
          for fpath in all_audio_filenames
      }
      segments['stem'] = segments['filename'].apply(
          lambda filename: filename.split('.')[0])

      # Log all segments that could not be matched to an actual audio file.
      audio_not_found = segments[segments['stem'].apply(
          lambda stem: stem not in stem2path)]
      logging.info('Audios that could not be found: %s.',
                   audio_not_found['stem'].unique())

      # Filter out all segments that could not be matched to an actual
      # audio file.
      segments = segments[segments['stem'].apply(
          lambda stem: stem in stem2path)]
      segments['url'] = segments.apply(
          lambda rec: audio_path / stem2path[rec['stem']], axis=1)
    else:
      # For unsupervised data, we have access to a set of non-annotated audio
      # files. Therefore, we collect them, and attach an "unknown" labelled
      # segment to each of the audio files.
      segments = pd.DataFrame(
          [x for x in all_audio_filenames if x.suffix in _AUDIO_EXTENSIONS],
          columns=['url'])
      segments['filename'] = segments['url'].apply(lambda x: x.stem)
      # For compatibility, we add an "unknown" annotation to the recording
      # dataframe that goes from start to end. That ensures that any interval
      # detected as signal by our localization function will appear in the
      # final audio set, with the 'unknown' annotation.
      segments['start_time_s'] = 0
      segments['end_time_s'] = -1
      for level in ['label', 'order', 'family', 'genus']:
        segments[level] = [['unknown'] for _ in range(len(segments))]
    logging.info('%s annotated segments detected', len(segments))
    return {
        'train': self._generate_examples(segments=segments),
    }

  def get_labeled_intervals(
      self,
      audio: np.ndarray,
      file_segments: pd.DataFrame,
  ) -> Dict[Tuple[int, int], Dict[str, Set[str]]]:
    """Slices the given audio, and produces labels intervals.

    `file_segments` corresponds to the segments annotated by recordists. The
    final intervals correspond to slices of the audio where actual signal
    is observed (according to the `slice_peaked_audio` function), and the
    corresponding labels correspond to the label from annotated segments which
    overlap with the slice.

    Args:
      audio: The full audio file, already loaded.
      file_segments: The annotated segments for this audio. Each row (=segment)
        must minimally contain the following fields: ['label', 'genus', 'order',
        'family', 'start_time_s', 'end_time_s'].

    Returns:
      labeled_intervals: A Dict mapping a (start, end) time of the recording to
      the sets of species/genus/orders/families present in that interval.
    """
    logging.info('Found %d annotations for target file.', len(file_segments))

    # Slice the audio into intervals
    sr = self.builder_config.sample_rate_hz
    # Returns `interval_length_s` long intervals.
    audio_intervals = [
        (int(st), int(end))
        for (st, end) in self.builder_config.localization_fn(
            audio, sr, self.builder_config.interval_length_s, max_intervals=200)
    ]
    interval_timestamps = sorted(audio_intervals)

    def _st_end_key(seg):
      if seg['end_time_s'] == -1:
        end = audio.shape[-1]
      else:
        end = int(sr * seg['end_time_s'])
        if seg['end_time_s'] < seg['start_time_s']:
          logging.warning(
              'Skipping annotated segment because end time is anterior to start time.'
          )
          return ()
      return (int(sr * seg['start_time_s']), end)

    # Search for intervals with annotations.
    segments_by_timestamp = {
        _st_end_key(seg): seg
        for _, seg in file_segments.iterrows()
        if _st_end_key(seg)
    }
    labeled_intervals = {}
    for (st, end) in interval_timestamps:
      interval_labels = collections.defaultdict(lambda: set([]))
      for (curr_anno_st, curr_anno_end), seg in segments_by_timestamp.items():
        # no overlap, interval < anno
        if end < curr_anno_st:
          continue
        # no overlap, interval > anno
        if curr_anno_end < st:
          continue
        # found an overlap!
        for level in ['label', 'order', 'family', 'genus']:
          for label in seg[level]:
            interval_labels[level].add(label)
      if interval_labels['label']:
        labeled_intervals[(st, end)] = interval_labels
    return labeled_intervals

  def _generate_examples(self, segments: pd.DataFrame):
    """Generate examples from the dataframe of segments.

    Args:
      segments: Dataframe of segments. Each row (=segment) must minimally
        contain the following fields ['filename', 'url', 'label', 'genus',
        'order', 'start_time_s', 'end_time_s'].
    """
    beam = tfds.core.lazy_imports.apache_beam
    librosa = tfds.core.lazy_imports.librosa

    def _process_group(
        segment_group: pd.DataFrame) -> List[Tuple[str, Dict[str, Any]]]:

      # Each segment in segment_group will generate a tf.Example. A lot of
      # fields, especially metadata ones will be shared between segments.
      # Therefore, we create a template.
      recording_template = segment_group.iloc[0].copy()

      # Load the audio associated with this group of segments
      url = recording_template['url']
      with tempfile.NamedTemporaryFile(mode='w+b', suffix=url.suffix) as f:
        f.write(url.read_bytes())
        # librosa outputs lots of warnings which we can safely ignore when
        # processing all Xeno-Canto files and PySoundFile is unavailable.
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          sr = self.builder_config.sample_rate_hz
          try:
            audio, _ = librosa.load(
                f.name, sr=sr, res_type=self.builder_config.resampling_method)
          except Exception as inst:
            logging.warning(
                'The audio at %s could not be loaded. Following'
                'exception occured: %s', url, inst)
            return []
          # We remove all short audios. These short audios are only observed
          # among caples_2020 unlabelled recordings.
          target_length = int(sr * self.builder_config.interval_length_s)
          if len(audio) < target_length:
            logging.warning('Skipping audio at %s because too short.', url)
            return []

          # Resampling can introduce artifacts that push the signal outside the
          # [-1, 1) interval.
          audio = np.clip(audio, -1.0, 1.0 - (1.0 / float(1 << 15)))

      labeled_intervals = self.get_labeled_intervals(audio, segment_group)

      # Remove all the fields we don't need from the recording_template. We set
      # errors='ignore' as some fields to be dropped may already not exist.
      recording_template = recording_template.drop(
          ['stem', 'url', 'start_time_s', 'end_time_s'],
          errors='ignore').to_dict()

      # Create a tf.Example for every segment.
      valid_segments = []
      for index, ((start, end),
                  segment_labels) in enumerate(labeled_intervals.items()):
        key = f"{recording_template['filename']}_{index}"
        valid_segments.append((key, {
            **recording_template,
            'label': list(segment_labels['label']),
            'genus': list(segment_labels['genus']),
            'family': list(segment_labels['family']),
            'order': list(segment_labels['order']),
            'audio': audio[start:end],
            'segment_start': start,
            'segment_end': end,
        }))
      return valid_segments

    pipeline = (
        beam.Create(group for _, group in segments.groupby('filename'))
        | beam.FlatMap(_process_group))
    return pipeline
