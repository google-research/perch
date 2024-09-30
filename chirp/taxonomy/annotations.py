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

"""Utilities for manipulating annotations."""

import csv
import dataclasses
from typing import Callable, Sequence

from etils import epath
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
    annotations: Sequence[TimeWindowAnnotation],
) -> pd.DataFrame:
  return pd.DataFrame.from_records(
      [dataclasses.asdict(anno) for anno in annotations]
  )


def write_annotations_csv(
    filepath: str | epath.Path,
    annotations: Sequence[TimeWindowAnnotation],
    label_separator: str = ' ',
) -> None:
  """Write annotations to a CSV file."""
  fieldnames = [f.name for f in dataclasses.fields(TimeWindowAnnotation)]
  fieldnames.remove('namespace')
  with epath.Path(filepath).open('w') as f:
    dr = csv.DictWriter(f, fieldnames)
    dr.writeheader()
    for anno in annotations:
      anno_dict = {f: getattr(anno, f) for f in fieldnames}
      anno_dict['label'] = label_separator.join(anno_dict['label'])
      dr.writerow(anno_dict)


def read_annotations_csv(
    annotations_filepath: epath.Path, namespace: str, label_separator: str = ' '
) -> Sequence[TimeWindowAnnotation]:
  """Read annotations as written by write_annotations_csv."""
  got_annotations = []
  with epath.Path(annotations_filepath).open('r') as f:
    dr = csv.DictReader(f)
    for row in dr:
      got_annotations.append(
          TimeWindowAnnotation(
              filename=row['filename'],
              namespace=namespace,
              start_time_s=float(row['start_time_s']),
              end_time_s=float(row['end_time_s']),
              label=row['label'].split(label_separator),
          )
      )
  return got_annotations


def read_dataset_annotations_csvs(
    filepaths: Sequence[epath.Path],
    filename_fn: Callable[[epath.Path, dict[str, str]], str],
    namespace: str,
    class_fn: Callable[[dict[str, str]], Sequence[str]],
    start_time_fn: Callable[[dict[str, str]], float],
    end_time_fn: Callable[[dict[str, str]], float],
    filter_fn: Callable[[dict[str, str]], bool] | None = None,
    delimiter: str = ',',
) -> Sequence[TimeWindowAnnotation]:
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
    with filepath.open('r') as f:
      reader = csv.DictReader(f, delimiter=delimiter)
      for row in reader:
        if filter_fn and filter_fn(row):
          continue
        filename = filename_fn(filepath, row)
        start = start_time_fn(row)
        end = end_time_fn(row)
        classes = class_fn(row)
        annotations.append(
            TimeWindowAnnotation(filename, start, end, namespace, classes)
        )
  return annotations
