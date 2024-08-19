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

"""Helpers for loading specific annotation CSVs."""

import json
import os

from chirp.taxonomy import annotations
from chirp.taxonomy import namespace_db
from etils import epath
import pandas as pd

_DEPRECATED2NEW = {
    'mallar': 'mallar3',
    'rufant1': 'rufant7',
}


def load_caples_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Loads the dataframe of all caples annotations from annotation CSV.

  Args:
    annotations_path: Filepath for the annotations CSV.

  Returns:
    DataFrame of annotations.
  """
  filename_fn = lambda _, row: row['fid'].strip()
  start_time_fn = lambda row: float(row['start_time_s'])
  end_time_fn = lambda row: float(row['end_time_s'])
  # Get rid of the one bad label in the dataset...
  filter_fn = lambda row: 'comros' in row['ebird_codes']
  class_fn = lambda row: row['ebird_codes'].split(' ')
  annos = annotations.read_dataset_annotations_csvs(
      [annotations_path],
      filename_fn=filename_fn,
      namespace='ebird2021',
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
  )
  segments = annotations.annotations_to_dataframe(annos)
  return segments


def load_cornell_annotations(
    annotations_path: epath.Path, file_id_prefix: str = ''
) -> pd.DataFrame:
  """Load the annotations from a Cornell Zenodo dataset."""
  start_time_fn = lambda row: float(row['Start Time (s)'])
  end_time_fn = lambda row: float(row['End Time (s)'])
  filter_fn = lambda row: False
  class_fn = lambda row: [  # pylint: disable=g-long-lambda
      row['Species eBird Code'].strip().replace('????', 'unknown')
  ]

  filename_fn = lambda filepath, row: file_id_prefix + row['Filename'].strip()
  annos = annotations.read_dataset_annotations_csvs(
      [annotations_path],
      filename_fn=filename_fn,
      namespace='ebird2021',
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
  )
  segments = annotations.annotations_to_dataframe(annos)
  return segments


def load_powdermill_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Load annotations from https://zenodo.org/records/4656848."""
  start_time_fn = lambda row: float(row['Begin Time (s)'])
  end_time_fn = lambda row: float(row['End Time (s)'])
  filter_fn = lambda row: False

  # Convert dataset labels to ebird2021.
  db = namespace_db.load_db()
  ebird_mapping = db.mappings['ibp2019_to_ebird2021']
  ebird_mapping_dict = ebird_mapping.mapped_pairs
  class_fn = lambda row: [  # pylint: disable=g-long-lambda
      ebird_mapping_dict.get(row['Species'].strip(), row['Species'].strip())
  ]

  annotation_filepaths = [annotations_path]
  filename_fn = lambda filepath, row: row['Filename'].strip()
  annos = annotations.read_dataset_annotations_csvs(
      annotation_filepaths,
      filename_fn=filename_fn,
      namespace=ebird_mapping.target_namespace,
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
  )
  segments = annotations.annotations_to_dataframe(annos)
  return segments


def load_weldy_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Loads annotations from https://zenodo.org/records/10252138."""
  filename_fn = lambda _, row: 'annotated_recordings/' + row['file'].strip()
  start_time_fn = lambda row: float(row['start'])
  end_time_fn = lambda row: float(row['end'])
  filter_fn = lambda row: False
  class_fn = lambda row: (  # pylint: disable=g-long-lambda
      row['label']
      .replace('unk', 'unknown')
      .replace('impossible', 'unknown')
      .replace('unknown_chip', 'unknown')
      .split(' ')
  )
  annos = annotations.read_dataset_annotations_csvs(
      [epath.Path(annotations_path)],
      filename_fn=filename_fn,
      namespace='weldy_calltype',
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
  )
  segments = annotations.annotations_to_dataframe(annos)
  return segments


def load_anuraset_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Loads raw audio annotations from https://zenodo.org/records/8342596."""
  filename_fn = lambda _, row: os.path.join(  # pylint: disable=g-long-lambda
      row['filename'].split('_')[0], row['filename'].strip()
  )
  start_time_fn = lambda row: float(row['start_time_s'])
  end_time_fn = lambda row: float(row['end_time_s'])
  # There are a few SPECIES_LALSE labels which according to the authors should
  # be ignored.
  filter_fn = lambda row: '_LALSE' in row['label']
  class_fn = lambda row: row['label'].split(' ')
  annos = annotations.read_dataset_annotations_csvs(
      [epath.Path(annotations_path)],
      filename_fn=filename_fn,
      namespace='anuraset',
      class_fn=class_fn,
      start_time_fn=start_time_fn,
      end_time_fn=end_time_fn,
      filter_fn=filter_fn,
  )
  segments = annotations.annotations_to_dataframe(annos)
  return segments


def load_reef_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Loads a dataframe of all annotations from the ReefSet JSON file.

  Args:
    annotations_path: path to dataset_v*.json.

  Returns:
    DataFrame of metadata parsed from the datasets
    Reef specific stuff:
    - All clips are 1.88sec long, so we fix all start and end times accordingly
    - We only take entries for which the dataset_type is sound_event_dataset, as
    other entries are only soundscape (habitat level) labels or just unlabeled
    completely
    - In future, should this add a header to the df that species the region
    somehow? Allowing selection by regional datasets
  """
  # Read the JSON file
  with annotations_path.open() as f:
    data = json.load(f)
  # Prepare a list of dictionaries for creating a DataFrame
  rows = []
  for entry in data:
    # Include only entries with "dataset_type": "sound_event_dataset"
    if entry.get('dataset_type') == 'sound_event_dataset':
      label = entry.get('label', '')
      # to use region.label format use:
      # label = f"{entry.get('region', '')}.{entry.get('label', '')}"
      row = {
          'filename': entry.get('file_name', ''),
          'start_time_s': 0.0,
          'end_time_s': 1.88,
          'namespace': 'reefs',
          'label': [label],
      }
      rows.append(row)
  # Create a DataFrame
  segments = pd.DataFrame(rows)
  return segments
