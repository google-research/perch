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

"""Utilities for data processing."""

import hashlib
import os.path
from typing import Any, Iterable, Sequence

import chirp.data.bird_taxonomy  # pylint: disable=unused-import
import chirp.data.soundscapes  # pylint: disable=unused-import
from chirp.preprocessing import pipeline as pipeline_
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
    RuntimeError: If no datasets are loaded.
  """
  if isinstance(dataset_directory, str):
    dataset_directory = [dataset_directory]

  if pipeline is None:
    raise ValueError(
        'data_utils.get_dataset() requires a valid initialized Pipeline object '
        'to be specified.'
    )
  read_config = tfds.ReadConfig(add_tfds_id=True)

  datasets = []
  dataset_info = None
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
  if dataset_info is None:
    raise RuntimeError('No datasets loaded.')
  return ds, dataset_info


def get_base_dataset(
    split: str,
    is_train: bool = False,
    dataset_directory: str = _DEFAULT_DATASET_DIR,
    tfds_data_dir: str | None = _DEFAULT_TFDS_DATADIR,
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

  Returns:
    The placeholder dataset.
  Raises:
    ValueError: If no initialized Pipeline is passed.
    RuntimeError: If no datasets are loaded.
  """
  read_config = tfds.ReadConfig(add_tfds_id=True)

  if tfds_data_dir:
    tfds.core.add_data_dir(tfds_data_dir)
    ds, dataset_info = tfds.load(
        dataset_directory,
        split=split,
        data_dir=tfds_data_dir,
        with_info=True,
        read_config=read_config,
        shuffle_files=is_train,
    )
  else:
    builder = tfds.builder_from_directory(dataset_directory)
    ds = builder.as_dataset(
        split=split, read_config=read_config, shuffle_files=is_train
    )
    dataset_info = builder.info

  return ds, dataset_info


def get_multi_dataset(
    split: str,
    dataset_directories: Sequence[str],
    pipelines: Sequence[pipeline_.Pipeline],
    final_pipeline: pipeline_.Pipeline,
    is_train: bool = False,
    tfds_data_dirs: Sequence[str] | None = None,
    tf_data_service_address: Any | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[tf.data.Dataset, tfds.core.DatasetInfo | None]:
  """Returns a single unified dataset from multiple datasets.

  Args:
    split: Data split, e.g. 'train', 'test', 'train[:80%]', etc.
    dataset_directories: List of dataset directories.
    pipelines: List of pipelines corresponding to each dataset directory.
    final_pipeline: A final pipeline to be applied to the unified ds
    is_train: If the dataset will be used for training.
    tfds_data_dirs: If provided, uses tfds.add_data_dir, and then tfds.load.
    tf_data_service_address: Address for TFDataService. Only used if is_train is
      set.
    weights: The probability distribution weights for each dataset. Defaults to
      a balanced distribution.

  Returns:
    A unified dataset.
    Dataset information (might need to decide which one if multiple datasets are
    used).
  """
  if tfds_data_dirs is None:
    tfds_data_dirs = [_DEFAULT_TFDS_DATADIR] * len(dataset_directories)
  num_datasets = len(dataset_directories)
  if len(pipelines) != num_datasets:
    raise ValueError('Length of pipelines does not match number of datasets.')
  if tfds_data_dirs is not None and len(tfds_data_dirs) != num_datasets:
    raise ValueError(
        'Length of tfds_data_dirs does not match number of datasets.'
    )
  if weights is not None and len(weights) != num_datasets:
    raise ValueError('Length of weights does not match number of datasets.')

  peek_datasets = []
  base_datasets = []
  dataset_infos = []
  for dataset_dir, ds_pipeline, tfds_data_dir in zip(
      dataset_directories, pipelines, tfds_data_dirs
  ):
    ds, dataset_info = get_base_dataset(
        split, is_train, dataset_dir, tfds_data_dir
    )
    base_datasets.append(ds)
    dataset_infos.append(dataset_info)
    # Construct the actual Pipeline object from the config
    ds = ds_pipeline(ds, dataset_info)
    peek_datasets.append(ds)

  # Unify Datasets by adding placeholders for missing features in each
  merge_op = pipeline_.AddTensorOp.from_datasets(peek_datasets)
  merge_datasets = []
  for base_dataset, pipeline, dataset_info in zip(
      base_datasets, pipelines, dataset_infos
  ):
    new_ops = tuple(pipeline.ops) + (merge_op,)
    new_pipeline = pipeline_.Pipeline(
        new_ops, pipeline.num_parallel_calls, pipeline.deterministic
    )
    ds = new_pipeline(base_dataset, dataset_info)
    merge_datasets.append(ds)

  # Create a new dataset object that interleaves the multiple datasets
  unified_ds = tf.data.Dataset.sample_from_datasets(
      merge_datasets, weights=weights
  )

  # Batch wants ds_info as an arg, but doesnt use it, so pass the last ds_info
  # TODO(benwilliamsgpt): allow Pipeline to take None as dataset_info
  unified_ds = final_pipeline(unified_ds, dataset_infos[-1])
  unified_ds = unified_ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Handle distributed data loading
  if is_train and tf_data_service_address:
    unified_ds = unified_ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
            job_name='chirp_job',
        )
    )
  return unified_ds, None


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
