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

"""Utility functions for manipulating soundscape data and annotations."""

import dataclasses
import os
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Set, Tuple

from absl import logging
from chirp.taxonomy import namespace
from chirp.taxonomy import namespace_db
from etils import epath
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tqdm

_AUDIO_EXTENSIONS = ['.flac', '.wav']
LocalizationFn = Callable[[Any, int, float, int], Sequence[Tuple[int, int]]]
MAX_INTERVALS_PER_FILE = 200


@dataclasses.dataclass
class MetadataFeature:
  """Data for handling a metadata feature.

  Attricbutes:
    source_key: Key used for the metadata in the original dataset.
    target_key: New key used for the feature in the output dataset.
    convert_fn: Function for parsing metadata feature from the original dataset.
      (For example, to convert strings in a CSV file to floats.)
    feature_type: TFDS feature type, which is used in the TFDS FeatureDict.
  """
  source_key: str
  target_key: str
  convert_fn: Callable[[str], Any]
  feature_type: tfds.features.tensor_feature.Tensor


MetadataLoaderType = Callable[[epath.Path, Dict[str, MetadataFeature]],
                              pd.DataFrame]


def load_class_list(class_list_name: str,
                    keep_unknown_annotation: bool) -> namespace.ClassList:
  """Loads the target class list, possibly adding an unknown label.

  Args:
    class_list_name: Name of the class list to load.
    keep_unknown_annotation: If True, add an 'unknown' class to the ClassList.

  Returns:
    The desired ClassList.
  """
  db = namespace_db.NamespaceDatabase.load_csvs()
  dataset_class_list = db.class_lists[class_list_name]

  if (keep_unknown_annotation and 'unknown' not in dataset_class_list.classes):
    # Create a new class list which includes the 'unknown' class.
    dataset_class_list = namespace.ClassList(
        dataset_class_list.name + '_unknown', dataset_class_list.namespace,
        ['unknown'] + list(dataset_class_list.classes))
  return dataset_class_list


def create_segments_df(
    all_audio_filepaths: Iterator[epath.Path],
    annotations_df: Optional[pd.DataFrame], supervised: bool,
    metadata_dir: epath.Path, metadata_fields: Dict[str, MetadataFeature],
    metadata_load_fn: Optional[MetadataLoaderType]) -> pd.DataFrame:
  """Create the dataframe of segments with annotations and audio urls.

  Args:
    all_audio_filepaths: Iterator for audio sources.
    annotations_df: DataFrame of annotations.TimeWindowAnnotation.
    supervised: Whether this is a supervised dataset.
    metadata_dir: Directory containing the dataset's metadata. Only considered
      if metadata_load_fn is provided.
    metadata_fields: Dictionary describing handling of metadata features.
    metadata_load_fn: Function for loading metadata.

  Returns:
    DataFrame of dataset annotations with metadata.

  """
  if supervised:
    # Combine segments with additional metadata (e.g Country).
    segments = combine_annotations_with_metadata(annotations_df, metadata_dir,
                                                 metadata_fields,
                                                 metadata_load_fn)
    logging.info('starting with %d annotations...', len(segments))
    segments = add_annotated_urls(segments, all_audio_filepaths)
  else:
    # For unsupervised data, we have access to a set of non-annotated audio
    # files. Therefore, we collect them, and attach an "unknown" labelled
    # segment to each of the audio files.
    segments = pd.DataFrame(all_audio_filepaths, columns=['url'])
    segments['filename'] = segments['url'].apply(lambda x: x.stem)
    # For compatibility, we add an "unknown" annotation to the recording
    # dataframe that goes from start to end. That ensures that any interval
    # detected as signal by our localization function will appear in the
    # final audio set, with the 'unknown' annotation.
    segments['start_time_s'] = 0
    segments['end_time_s'] = -1
    segments['label'] = [['unknown'] for _ in range(len(segments))]
  logging.info('%s annotated segments detected', len(segments))
  return segments


def combine_annotations_with_metadata(
    segments: pd.DataFrame,
    metadata_dir: epath.Path,
    metadata_fields: Dict[str, MetadataFeature],
    metadata_load_fn: Optional[MetadataLoaderType],
    metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
  """Combine segments with whatever metadata is available for this dataset.

  Args:
    segments: DataFrame of annotations.TimeWindowAnnotation
    metadata_dir: Directory containing the dataset's metadata. Only considered
      if metadata_load_fn is provided.
    metadata_fields: Dictionary describing handling of metadata features.
    metadata_load_fn: Function for loading metadata.
    metadata_df: DataFrame of pre-loaded metadata. (testing convenience.)

  Returns:
    DataFrame of joined annotations and metadata.
  """
  if metadata_load_fn is None:
    return segments

  if metadata_df is None:
    # Load the dataframe containing the metadata. Each row describes some audio
    # file, and the dataframe should contain the 'filename' column, which acts
    # as the key to match with segments.
    metadata_df = metadata_load_fn(metadata_dir, metadata_fields)
  fid_to_metadata_index = metadata_df.groupby('filename').groups
  combined_segments = []
  bar = tqdm.tqdm(segments.iterrows(), total=len(segments))
  bar.set_description('Combining segments will full metadata.')
  for _, segment in bar:
    fid = segment['filename']
    segment_metadata = metadata_df.loc[fid_to_metadata_index[fid]]
    if segment_metadata.empty:
      logging.warning('MediaId %d not found in metadata', fid)

    for field in metadata_fields.values():
      if field.target_key == 'filename':
        # filename is special and we don't want to overwrite it.
        continue
      segment[field.target_key] = field.convert_fn(
          segment_metadata[field.target_key].iloc[0])
    combined_segments.append(segment)
  concat_segments = pd.concat(combined_segments, axis=1).T
  return concat_segments


def add_annotated_urls(
    segments: pd.DataFrame,
    all_audio_filepaths: Iterator[epath.Path]) -> pd.DataFrame:
  """Creates URLs for annotated segments, matching them to audio files.

  Args:
    segments: DataFrame of annotations and metadata.
    all_audio_filepaths: Iterator for audio sources.

  Returns:
    Updated segments DataFrame with URL's for existent audio sources.
  Raises:
    ValueError if no URLs are found.
  """
  # Our strategy is to match file stems, while checking that there
  # are no collisions. This works for all known soundscape datasets,
  # which typically have very structured filenames even if there are
  # multiple levels of file organization.
  stem_to_filepath = {}
  for fp in all_audio_filepaths:
    stem = fp.stem.split('.')[0]
    if stem in stem_to_filepath:
      raise ValueError('Found two files (%s vs %s) with the same stem.' %
                       (fp, stem_to_filepath[stem]))
    stem_to_filepath[stem] = fp

  segments['stem'] = segments['filename'].apply(
      lambda filename: os.path.basename(filename).split('.')[0])
  # Log all segments that could not be matched to an actual audio file.
  audio_not_found = segments[segments['stem'].apply(
      lambda stem: stem not in stem_to_filepath)]
  logging.info('Audios that could not be found: %s.',
               audio_not_found['stem'].unique())

  segments['url'] = segments.apply(
      lambda rec: stem_to_filepath.get(rec['stem'], ''), axis=1)
  # Filter segments without urls.
  segments = segments[segments['url'].apply(lambda url: url != '')]  # pylint: disable=g-explicit-bool-comparison
  if segments.empty:
    raise ValueError('No segments found. Likely a problem matching '
                     'annotation filenames to audio.')
  segments = segments.drop('stem', axis=1)
  return segments


def get_labeled_intervals(
    audio: np.ndarray,
    file_segments: pd.DataFrame,
    class_list: namespace.ClassList,
    sample_rate_hz: int,
    interval_length_s: int,
    localization_fn: LocalizationFn,
) -> Dict[Tuple[int, int], Set[str]]:
  """Slices the given audio, and produces labels intervals.

  `file_segments` corresponds to the segments annotated by recordists. The
  final intervals correspond to slices of the audio where actual signal
  is observed (according to the `slice_peaked_audio` function), and the
  corresponding labels correspond to the label from annotated segments which
  overlap with the slice.

  Args:
    audio: The full audio file, already loaded.
    file_segments: The annotated segments for this audio. Each row (=segment)
      must minimally contain the following fields: ['label', 'start_time_s',
      'end_time_s'].
    class_list: List of labels which will appear in the processed dataset.
    sample_rate_hz: Sample rate of audio.
    interval_length_s: Window size to slice.
    localization_fn: Function for selecting audio intervals.

  Returns:
    labeled_intervals: A Dict mapping a (start, end) time of the recording to
    the set of classes present in that interval.
  """
  logging.info('Found %d annotations for target file.', len(file_segments))

  # Slice the audio into intervals
  # Returns `interval_length_s` long intervals.
  audio_intervals = [(int(st), int(end)) for (st, end) in localization_fn(
      audio, sample_rate_hz, interval_length_s, MAX_INTERVALS_PER_FILE)]
  interval_timestamps = sorted(audio_intervals)

  def _start_end_key(seg):
    if seg['end_time_s'] == -1:
      end = audio.shape[-1]
    else:
      end = int(sample_rate_hz * seg['end_time_s'])
      if seg['end_time_s'] < seg['start_time_s']:
        logging.warning(
            'Skipping annotated segment because end time is anterior to start '
            'time.')
        return ()
    return (int(sample_rate_hz * seg['start_time_s']), end)

  # Search for intervals with annotations.
  segments_by_timestamp = {
      _start_end_key(seg): seg
      for _, seg in file_segments.iterrows()
      if _start_end_key(seg)
  }
  labeled_intervals = {}
  for (st, end) in interval_timestamps:
    interval_labels = set([])
    for (current_annotation_start,
         currrent_annotation_end), seg in segments_by_timestamp.items():
      # no overlap, interval < anno
      if end < current_annotation_start:
        continue
      # no overlap, interval > anno
      if currrent_annotation_end < st:
        continue
      # found an overlap!
      for label in seg['label']:
        if label and label not in class_list.classes:
          logging.warning('Found label "%s" not in the dataset classlist.',
                          label)
          continue
        if label:
          interval_labels.add(label)
    if interval_labels:
      labeled_intervals[(st, end)] = interval_labels
  return labeled_intervals
