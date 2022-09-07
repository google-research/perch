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

import dataclasses
import tempfile
from typing import Callable, Dict, Optional, Tuple, List, Any
import warnings

from absl import logging
from chirp import audio_utils
from chirp.data.bird_taxonomy import bird_taxonomy
from chirp.data.soundscapes import dataset_fns
from chirp.data.soundscapes import soundscapes_lib
from etils import epath
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

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


@dataclasses.dataclass
class SoundscapesConfig(bird_taxonomy.BirdTaxonomyConfig):
  """Dataset configuration for Soundscape datasets.

  Attributes:
    audio_glob: Pattern to match to find audio files.
    class_list_name: The name of the ClassList to use for labels. This is
      typically a list of either all regionally feasible species in the area
      (for fully-annotated datasets) or the list of all species annotated
      (if only a subset has been labeled).
    metadata_load_fn: Because the metadata don't always appear under the same
      format, we specify for each config the way to load metadata. This function
      outputs a dataframe, where each row contains the metadata for some audio
      file. The column names of the dataframe should match the keys in
      metadata_fields.
    metadata_fields: Maps the fields of the metadata DataFrame to tfds.features
      datatypes.
    annotation_load_fn: Because the annotations don't always appear in the same
      format, we specify a function to load the annotations.
    keep_unknown_annotation: An "unknown" annotations appears in some datasets.
      This boolean decides whether it should keep this annotation (and
      therefore) add a species named "unknown" in the label space, or just scrub
      all "unknown" annotations.
  """
  audio_glob: str = ''
  class_list_name: str = ''
  metadata_load_fn: Optional[soundscapes_lib.MetadataLoaderType] = None
  metadata_fields: Optional[Dict[str, soundscapes_lib.MetadataFeature]] = None
  annotation_load_fn: Optional[Callable[[epath.Path], pd.DataFrame]] = None
  keep_unknown_annotation: bool = False
  supervised: bool = True
  audio_dir = epath.Path('gs://chirp-public-bucket/soundscapes')


class Soundscapes(bird_taxonomy.BirdTaxonomy):
  """DatasetBuilder for soundscapes data."""

  VERSION = tfds.core.Version('1.0.3')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release. The label set corresponds to the full '
               'set of ~11 000 Xeno-Canto species.',
      '1.0.1': 'The label set is now restricted to the species present in each'
               'dataset.',
      '1.0.2': 'Streamlines data handling, and adds handling for a new '
               'Sapsucker Woods dataset.',
      '1.0.3': 'Adds handling for the new Cornell Sierra Nevadas dataset and '
               'the Kitzeslab Powdermill dataset.',
  }
  BUILDER_CONFIGS = [
      # pylint: disable=unexpected-keyword-arg
      SoundscapesConfig(
          name='caples',  # TODO(mboudiaf) Try to interface caples metadata.
          class_list_name='caples',
          audio_glob='caples/audio/*',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_caples_annotations,
          description=('Annotated Caples recordings from 2018/2019.')),
      SoundscapesConfig(
          name='hawaii',
          audio_glob='hawaii/*/*.wav',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_hawaii_annotations,
          description=('Fully annotated Hawaii recordings.'),
          class_list_name='hawaii'),
      SoundscapesConfig(
          name='ssw',
          audio_glob='ssw/audio/*.flac',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_ssw_annotations,
          description=('Annotated Sapsucker Woods recordings. '
                       'https://zenodo.org/record/7018484'),
          class_list_name='new_york'),
      SoundscapesConfig(
          name='birdclef2019_colombia',
          audio_glob='birdclef2019/audio/*.wav',
          metadata_load_fn=dataset_fns.load_birdclef_metadata,
          metadata_fields=dataset_fns.birdclef_metadata_features(),
          annotation_load_fn=dataset_fns.load_birdclef_annotations,
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          description=(
              'Colombian recordings from the Birdclef 2019 challenge.'),
          class_list_name='birdclef2019_colombia'),
      SoundscapesConfig(
          name='high_sierras',
          audio_glob='high_sierras/audio/*.wav',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_birdclef_annotations,
          description=('High Sierras recordings.'),
          class_list_name='high_sierras'),
      SoundscapesConfig(
          name='sierras_kahl',
          audio_glob='sierras_kahl/audio/*.flac',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_sierras_kahl_annotations,
          description=('Sierra Nevada recordings. '
                       'https://zenodo.org/record/7050014'),
          class_list_name='sierras_kahl'),
      SoundscapesConfig(
          name='powdermill',
          audio_glob='powdermill/*/*.flac',
          interval_length_s=5.0,
          localization_fn=audio_utils.slice_peaked_audio,
          annotation_load_fn=dataset_fns.load_powdermill_annotations,
          description=('New England recordings from Powdermill Nature Reserve, '
                       'Rector, PA. https://doi.org/10.1002/ecy.3329'),
          class_list_name='powdermill'),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    dataset_class_list = soundscapes_lib.load_class_list(
        self.builder_config.class_list_name,
        self.builder_config.keep_unknown_annotation)
    logging.info('Currently considering a total of %s species.',
                 dataset_class_list.size)
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
                tfds.features.ClassLabel(names=dataset_class_list.classes)),
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
          k.target_key: k.feature_type
          for k in self.builder_config.metadata_fields.values()
      }
      common_features.update(additional_features)
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(common_features),
        supervised_keys=('audio', 'label'),
        homepage='https://github.com/google-research/chirp',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # Defined as part of the tfds API for dividing the dataset into splits.
    # https://www.tensorflow.org/datasets/add_dataset#specifying_dataset_splits
    dl_manager._force_checksums_validation = False  # pylint: disable=protected-access

    # Get the state from the dl_manager which we'll use to create segments.
    all_audio_filenames = self.builder_config.audio_dir.glob(
        self.builder_config.audio_glob)
    if self.builder_config.supervised:
      # For supervised data, we first grab the annotated segments.
      annotations_path = dl_manager.download_and_extract({
          'segments': (self.builder_config.audio_dir / 'metadata' /
                       f'{self.builder_config.name}.csv').as_posix(),
      })['segments']
      annotations_df = self.builder_config.annotation_load_fn(annotations_path)
    else:
      annotations_df = None

    segments = soundscapes_lib.create_segments_df(
        all_audio_filenames, annotations_df, self.builder_config.supervised,
        self.builder_config.audio_dir, self.builder_config.metadata_fields,
        self.builder_config.metadata_load_fn)
    return {
        'train': self._generate_examples(segments=segments),
    }

  def _generate_examples(self, segments: pd.DataFrame):
    """Generate examples from the dataframe of segments.

    Args:
      segments: Dataframe of segments. Each row (=segment) must minimally
        contain the following fields:
        ['filename', 'url', 'label', 'start_time_s', 'end_time_s'].
    Returns:
      List of valid segments.
    """
    beam = tfds.core.lazy_imports.apache_beam
    librosa = tfds.core.lazy_imports.librosa
    info = self._info()
    # Drop any extraneous columns.
    for k in segments.columns.values:
      if (k not in info.features and
          k not in ['url', 'start_time_s', 'end_time_s']):
        segments = segments.drop(k, axis=1)

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
          except Exception as inst:  # pylint: disable=broad-except
            # We have no idea what can go wrong in librosa, so we catch a braod
            # exception here.
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

      class_list = soundscapes_lib.load_class_list(
          self.builder_config.class_list_name,
          self.builder_config.keep_unknown_annotation)
      labeled_intervals = soundscapes_lib.get_labeled_intervals(
          audio, segment_group, class_list, self.builder_config.sample_rate_hz,
          self.builder_config.interval_length_s,
          self.builder_config.localization_fn)

      # Remove all the fields we don't need from the recording_template. We set
      # errors='ignore' as some fields to be dropped may already not exist.
      recording_template = recording_template.drop(
          ['url', 'start_time_s', 'end_time_s'], errors='ignore').to_dict()

      # Create a tf.Example for every segment.
      valid_segments = []
      for index, ((start, end),
                  segment_labels) in enumerate(labeled_intervals.items()):
        key = f"{recording_template['filename']}_{index}"
        valid_segments.append((key, {
            **recording_template,
            'label': list(segment_labels),
            'audio': audio[start:end],
            'segment_start': start,
            'segment_end': end,
        }))
      return valid_segments

    pipeline = (
        beam.Create(group for _, group in segments.groupby('filename'))
        | beam.FlatMap(_process_group))
    return pipeline
