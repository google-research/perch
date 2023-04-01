# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Utilities for data processing."""

import hashlib
import os.path
from typing import Any, Iterable

from chirp import preprocessing as pipeline_
import chirp.data.bird_taxonomy  # pylint: disable=unused-import
import chirp.data.soundscapes  # pylint: disable=unused-import
import tensorflow as tf
import tensorflow_datasets as tfds


# Import bird_taxonomy and soundscapes to register the datasets with TFDS.
_DEFAULT_DATASET_DIR = None
_DEFAULT_TFDS_DATADIR = None
_DEFAULT_PIPELINE = None


def get_dataset(
    split: str,
    is_train: bool = False,
    dataset_directory: str | Iterable[str] = _DEFAULT_DATASET_DIR,
    tfds_data_dir: str | None = _DEFAULT_TFDS_DATADIR,
    tf_data_service_address: Any | None = None,
    pipeline: pipeline_.Pipeline | None = _DEFAULT_PIPELINE,
) -> tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
  """Returns the placeholder dataset.

  Args:
    split: data split, e.g. 'train', 'test', 'train[:80%]', etc.
    is_train: If the dataset will be used for training. This only affects
      whether data will be distributed or not in case tf_data_service_address is
      provided.
    dataset_directory: dataset directory. If multiple are passed, then samples
      are uniformly taken from each dataset. When multiple datasets are loaded,
      only the dataset info of the first dataset is returned.
    tfds_data_dir: If provided, uses tfds.add_data_dir, and then tfds.load,
      instead of using the tfds.builder_from_directory.
    tf_data_service_address: Address for TFDataService. Only used if is_train is
      set.
    pipeline: (required) A preprocessing pipeline to apply to the data.

  Returns:
    The placeholder dataset.
  Raises:
    ValueError: If no initialized Pipeline is passed.
  """
  if isinstance(dataset_directory, str):
    dataset_directory = [dataset_directory]

  if pipeline is None:
    raise ValueError(
        'requires a valid initialized Pipeline object to be specified.'
    )
  read_config = tfds.ReadConfig(add_tfds_id=True)

  datasets = []
  for dataset_dir in dataset_directory:
    if tfds_data_dir:
      tfds.core.add_data_dir(tfds_data_dir)
      ds, dataset_info = tfds.load(
          dataset_dir,
          split=split,
          data_dir=tfds_data_dir,
          with_info=True,
          read_config=read_config,
          shuffle_files=is_train,
      )
    else:
      builder = tfds.builder_from_directory(dataset_dir)
      ds = builder.as_dataset(
          split=split, read_config=read_config, shuffle_files=is_train
      )
      dataset_info = builder.info

    datasets.append(pipeline(ds, dataset_info))

  if len(datasets) > 1:
    ds = tf.data.Dataset.sample_from_datasets(datasets)
  else:
    ds = datasets[0]

  if is_train and tf_data_service_address:
    ds = ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
            job_name='chirp_job',
        )
    )
  ds = ds.prefetch(2)
  return ds, dataset_info


def xeno_canto_filename(filename: str, id_: int) -> tuple[str, str]:
  """Determine a filename for a Xeno-Canto recording.

  We can't use the original filename since some of those are not valid Unix
  filenames (e.g., they contain slashes). Hence the files are named using just
  their ID. There are some files with spaces in the extension, so that is
  handled here as well.

  We also return the first two characters of the MD5 hash of the filename. This
  can be used to evenly distribute files across 256 directories in a
  deterministic manner.

  Args:
    filename: The original filename (used to determine the extension).
    id_: The Xeno-Canto ID of the recording.

  Returns:
    A tuple where the first element is the filename to save this recording two
    and the second element is a two-character subdirectory name in which to
    save the file.
  """
  # Two files have the extension ". mp3"
  ext = os.path.splitext(filename)[1].lower().replace(' ', '')
  filename = f'XC{id_}{ext}'
  # Results in ~2900 files per directory
  subdir = hashlib.md5(filename.encode()).hexdigest()[:2]
  return filename, subdir
