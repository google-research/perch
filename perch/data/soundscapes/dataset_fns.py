# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Config utils specific to BirdClef Soundscape datasets."""

import csv
import os

from chirp.data.soundscapes import soundscapes_lib
from chirp.taxonomy import annotations
from chirp.taxonomy import namespace_db
from etils import epath
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

_DEPRECATED2NEW = {
    'mallar': 'mallar3',
    'rufant1': 'rufant7',
}


def load_birdclef_metadata(
    root: epath.Path,
    metadata_feature_info: dict[str, soundscapes_lib.MetadataFeature],
) -> pd.DataFrame:
  """The `metadata_load_fn` for Birdclef2019-based configs.

  Args:
    root: Base dataset path.
    metadata_feature_info: Dictionary describing the desired metadata features.

  Returns:
    DataFrame of metadata parsed from the dataset.
  """
  metadata_path = root / 'birdclef2019' / 'metadata'
  df = []
  bar = tqdm.tqdm(metadata_path.iterdir())
  bar.set_description('Loading BirdClef2019 metadata.')
  for path in bar:
    with path.open('rb') as f:
      df.append(pd.read_json(f, typ='series'))
  df = pd.concat(df, axis=1).T
  for feature in metadata_feature_info.values():
    df[feature.target_key] = df[feature.source_key].map(feature.convert_fn)
    df = df.drop(feature.source_key, axis=1)
  return df


def birdclef_metadata_features() -> dict[str, soundscapes_lib.MetadataFeature]:
  """Metadata features to join with BirdClef data."""
  feature_types = {
      'filename': soundscapes_lib.MetadataFeature(
          'FileName', 'filename', str, tfds.features.Text()
      ),
      'country': soundscapes_lib.MetadataFeature(
          'Country', 'country', str, tfds.features.Text()
      ),
      'longitude': soundscapes_lib.MetadataFeature(
          'Longitude',
          'longitude',
          float,
          tfds.features.Scalar(dtype=tf.float32),
      ),
      'latitude': soundscapes_lib.MetadataFeature(
          'Latitude', 'latitude', float, tfds.features.Scalar(dtype=tf.float32)
      ),
      'elevation': soundscapes_lib.MetadataFeature(
          'Elevation',
          'elevation',
          float,
          tfds.features.Scalar(dtype=tf.float32),
      ),
      'recordist': soundscapes_lib.MetadataFeature(
          'AuthorID', 'recordist', str, tfds.features.Text()
      ),
  }
  return feature_types


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


def load_cornell_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Load the annotations from a Cornell Zenodo dataset."""
  start_time_fn = lambda row: float(row['Start Time (s)'])
  end_time_fn = lambda row: float(row['End Time (s)'])
  filter_fn = lambda row: False
  class_fn = lambda row: [  # pylint: disable=g-long-lambda
      row['Species eBird Code'].strip().replace('????', 'unknown')
  ]

  filename_fn = lambda filepath, row: row['Filename'].strip()
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


# TODO(tomdenton): Eliminate these 'combine' functions.
# Reading directly from the set of annotation files will be more direct and
# less error prone when updating datasets.
def combine_powdermill_annotations(
    dataset_path: epath.Path, output_filepath: epath.Path
) -> None:
  """Combine all Powdermill dataset annotations into a single csv."""
  tables = dataset_path.glob('*/*.txt')
  fieldnames = [
      'Selection',
      'View',
      'Channel',
      'Begin Time (s)',
      'End Time (s)',
      'High Freq (Hz)',
      'Low Freq (Hz)',
      'Species',
  ]
  rows = []
  for table_fp in tables:
    with table_fp.open('r') as f:
      reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
      subdir_name = table_fp.parent.name
      audio_filename = os.path.basename(table_fp).split('.')[0] + '.wav'
      for row in reader:
        # Some annotation files have a header, and some do not.
        # So we skip the headers when present.
        if row['View'] == 'View':
          continue
        # The filename in the row doesn't include the file's directory.
        row['Filename'] = os.path.join(subdir_name, audio_filename)
        rows.append(row)

  with output_filepath.open('w') as f:
    fieldnames.append('Filename')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def load_powdermill_annotations(annotations_path: epath.Path) -> pd.DataFrame:
  """Load the dataframe of all Powdermill annotations from annotation CSV."""
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
  """Loads a dataframe of all annotations from the Weldy Calltype dataset."""
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
  """Loads a dataframe of all annotations."""
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
