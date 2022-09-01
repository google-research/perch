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

"""Utilities for manipulating annotations."""

import csv
import dataclasses
import glob
import os
from typing import Callable, Dict, Optional, Sequence

from chirp.taxonomy import namespace_db
import pandas as pd


@dataclasses.dataclass
class TimeWindowAnnotation:
  """An annotation for a particular time window.

  Attributes:
    filename: Filename for the source audio.
    start_time_s: Float representing the start of this annotation window.
    end_time_s: Float representing the end of this annotation window.
    namespace: The namespace of the classes in this annotation.
    label: List of classes present in the audio segment.
  """
  filename: str
  start_time_s: float
  end_time_s: float
  namespace: str
  label: Sequence[str]


def annotations_to_dataframe(
    annotations: Sequence[TimeWindowAnnotation]) -> pd.DataFrame:
  return pd.DataFrame.from_records(
      [dataclasses.asdict(anno) for anno in annotations])


def write_annotations_csv(filepath, annotations):
  fieldnames = [f.name for f in dataclasses.fields(TimeWindowAnnotation)]
  fieldnames.remove('namespace')
  with open(filepath, 'w') as f:
    dr = csv.DictWriter(f, fieldnames)
    dr.writeheader()
    for anno in annotations:
      anno_dict = {f: getattr(anno, f) for f in fieldnames}
      anno_dict['label'] = ' '.join(anno_dict['label'])
      dr.writerow(anno_dict)


def read_dataset_annotations_csvs(
    filepaths: Sequence[str],
    filename_fn: Callable[[str, Dict[str, str]], str],
    namespace: str,
    class_fn: Callable[[Dict[str, str]], Sequence[str]],
    start_time_fn: Callable[[Dict[str, str]], float],
    end_time_fn: Callable[[Dict[str, str]], float],
    filter_fn: Optional[Callable[[Dict[str, str]], bool]] = None,
    delimiter: str = ',') -> Sequence[TimeWindowAnnotation]:
  """Create annotations from a random CSV.

  Args:
    filepaths: Path to the CSV files.
    filename_fn: Function for extracting the audio filename. Maps
      (annotations_filename, row) to the filename of the audio.
    namespace: Namespace for the annotated classes.
    class_fn: Function for extracting classname.
    start_time_fn: Field for starting timestamps. Currently assumes values are
      floats measured in seconds.
    end_time_fn: Field for ending timestamps.
    filter_fn: A function for selecting rows of the annotation file to ignore.
      Will keep rows where filter_fn is False, and ignore rows where True.
    delimiter: Field separating character in the target file.

  Returns:
    List of TimeWindowAnnotations.
  """
  annotations = []
  for filepath in filepaths:
    with open(filepath, 'r') as f:
      reader = csv.DictReader(f, delimiter=delimiter)
      for row in reader:
        if filter_fn and filter_fn(row):
          continue
        filename = filename_fn(filepath, row)
        start = start_time_fn(row)
        end = end_time_fn(row)
        classes = class_fn(row)
        annotations.append(
            TimeWindowAnnotation(filename, start, end, namespace, classes))
  return annotations


#   +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
#  /  Dataset-specific annotation handling. /
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+


def combine_hawaii_dataset_annotations(dataset_path: str, output_filepath: str):
  """Combine all Hawaii dataset annotations into a single csv."""
  tables = glob.glob(os.path.join(dataset_path, '*', '*.txt'))
  rows = []
  for table_fp in tables:
    with open(table_fp) as f:
      reader = csv.DictReader(f, delimiter='\t')
      for row in reader:
        # The filename in the row doesn't include the directory.
        row['Begin File'] = os.path.join(
            os.path.basename(os.path.dirname(table_fp)), row['Begin File'])
        # The Spectrogram rows are redundant with the Waveform rows, but contain
        # a couple extra columns.
        if 'Spectrogram' not in row['View']:
          rows.append(row)

  with open(output_filepath, 'w') as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def read_hawaii_dataset_annotations(
    dataset_path: str,
    use_combined_csv=False) -> Sequence[TimeWindowAnnotation]:
  """Read annotations from Raven Selection Tables for the Hawaii dataset."""
  start_time_fn = lambda row: float(row['Begin Time (s)'])
  end_time_fn = lambda row: float(row['End Time (s)'])
  filter_fn = lambda row: 'Spectrogram' in row['View']

  # Convert dataset labels to ebird2021.
  db = namespace_db.NamespaceDatabase.load_csvs()
  ebird_mapping = db.mappings['hawaii_dataset_to_ebird2021'].to_dict()
  class_fn = lambda row: [ebird_mapping.get(row['Species'].strip(), '')]

  if use_combined_csv:
    annotation_filepaths = [dataset_path]
    filename_fn = lambda filepath, row: row['Begin File'].strip()
    delimiter = ','
  else:
    annotation_filepaths = glob.glob(os.path.join(dataset_path, '*/*.txt'))
    # Audio files are in per-island subdirectories.
    dir_name = lambda filepath: os.path.basename(os.path.dirname(filepath))
    filename_fn = lambda filepath, row: os.path.join(  # pylint: disable=g-long-lambda
        dir_name(filepath), row['Begin File'].strip())
    delimiter = '\t'

  annotations = read_dataset_annotations_csvs(
      annotation_filepaths,
      filename_fn=filename_fn,
      namespace='ebird2021',
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
      delimiter=delimiter)
  return annotations


def read_ssw_dataset_annotations(
    annotations_path: str) -> Sequence[TimeWindowAnnotation]:
  """Read annotations from Raven Selection Tables for the SSW dataset."""
  start_time_fn = lambda row: float(row['Start Time (s)'])
  end_time_fn = lambda row: float(row['End Time (s)'])
  filter_fn = lambda row: False
  # SSW data are already using ebird codes.
  class_fn = lambda row: [row['Species eBird Code'].strip()]
  filename_fn = lambda filepath, row: row['Filename'].strip()

  annotations = read_dataset_annotations_csvs([annotations_path],
                                              filename_fn=filename_fn,
                                              namespace='ebird2021',
                                              class_fn=class_fn,
                                              start_time_fn=start_time_fn,
                                              end_time_fn=end_time_fn,
                                              filter_fn=filter_fn)
  return annotations


def read_caples_dataset_annotations(
    dataset_path: str) -> Sequence[TimeWindowAnnotation]:
  """Read annotations from ornithology2 annotations of the Caples data."""
  annotations_path = os.path.join(dataset_path, 'caples.csv')
  filename_fn = lambda _, row: row['fid'].strip() + '.wav'
  start_time_fn = lambda row: float(row['start_time_s'])
  end_time_fn = lambda row: float(row['end_time_s'])
  # Get rid of the one bad label in the dataset...
  filter_fn = lambda row: 'comros' in row['ebird_codes']
  class_fn = lambda row: ' '.split(row['ebird_codes'].strip())
  return read_dataset_annotations_csvs([annotations_path],
                                       filename_fn=filename_fn,
                                       namespace='ebird2021',
                                       class_fn=class_fn,
                                       start_time_fn=start_time_fn,
                                       end_time_fn=end_time_fn,
                                       filter_fn=filter_fn)


def read_birdclef_dataset_annotations(
    annotations_path: str) -> Sequence[TimeWindowAnnotation]:
  """Read annotations from ornithology2 annotations in a single csv file."""
  filename_fn = lambda _, row: row['fid'].strip() + '.wav'
  start_time_fn = lambda row: float(row['start_time_s'])
  end_time_fn = lambda row: float(row['end_time_s'])
  filter_fn = lambda row: row['end_time_s'] <= row['start_time_s']
  class_fn = lambda row: ' '.split(row['ebird_codes'].strip())
  return read_dataset_annotations_csvs([annotations_path],
                                       filename_fn=filename_fn,
                                       namespace='ebird2021',
                                       class_fn=class_fn,
                                       start_time_fn=start_time_fn,
                                       end_time_fn=end_time_fn,
                                       filter_fn=filter_fn)
