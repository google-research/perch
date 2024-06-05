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

"""Utilities for training separated clustering.

The core workflow consists of the following:
a) Make a collection of wav files divided by label into sub-directories.
b) Load an `interface.EmbeddingModel`.
c) Create a MergedDataset using the directory and embedding model.
  This will load all of the labeled wavs and run the embedding model over
  all of them, creating an in-memory dataset.

This dataset can then be used for all kinds of small experiments, such as
training small classifiers or evaluating clustering methods.
"""

import collections
import dataclasses
import itertools
import time
from typing import Dict, Sequence, Tuple

from chirp import audio_utils
from chirp.inference import interface
from chirp.inference import tf_examples
from etils import epath
import numpy as np
import tensorflow as tf
import tqdm


@dataclasses.dataclass
class MergedDataset:
  """In-memory dataset of labeled audio with embeddings.

  Attributes:
    data: Dictionary of embedding outputs.
    num_classes: Number of classes.
    embedding_dim: Dimension of embeddings.
    labels: Tuple with the labels for each file.
  """

  # The following are populated automatically from one of two classmethods.
  data: Dict[str, np.ndarray]
  num_classes: int
  embedding_dim: int
  labels: Tuple[str, ...]

  @classmethod
  def from_folder_of_folders(
      cls,
      base_dir: str,
      embedding_model: interface.EmbeddingModel,
      time_pooling: str = 'mean',
      exclude_classes: Sequence[str] = (),
      load_audio: bool = False,
      target_sample_rate: int = -2,
      audio_file_pattern: str = '*',
      embedding_config_hash: str = '',
      embedding_file_prefix: str = 'embeddings-',
      pad_type: str = 'zeros',
      cache_embeddings: bool = True,
      tf_record_shards: int = 1,
      max_workers: int = 5,
  ) -> 'MergedDataset':
    """Generating MergedDataset via folder-of-folders method.

    This method will scan for existing embeddings cached within the folder of
    folders and re-use those with a matching prefix. The prefix is expected to
    have a hash signature for matching configs.

    Args:
      base_dir: Base directory where either folder-of-folders of audio or
        tfrecord embeddings are stored.
      embedding_model: EmbeddingModel used to produce embeddings.
      time_pooling: Key for time pooling strategy.
      exclude_classes: Classes to skip.
      load_audio: Whether to load audio into memory. beware that this can cause
        problems with large datasets.
      target_sample_rate: Resample loaded audio to this sample rate. If -1,
        loads raw audio with no resampling. If -2, uses the embedding_model
        sample rate.
      audio_file_pattern: The glob pattern to use for finding audio files within
        the sub-folders.
      embedding_config_hash: String hash of the embedding config to identify an
        existing embeddings folder. This will be appended to the
        embedding_file_prefix, e.g. 'embeddings-1234'.
      embedding_file_prefix: Prefix for existing materialized embedding files.
        Embeddings with a matching hash will be re-used to avoid reprocessing,
        and embeddings with a non-matching hash will be ignored.
      pad_type: Padding strategy for short audio.
      cache_embeddings: Materialize new embeddings as TF records within the
        folder-of-folders.
      tf_record_shards: Number of files to materialize if writing embeddings to
        TF records.
      max_workers: Number of threads to use for loading audio.

    Returns:
      MergedDataset
    """
    print('Embedding from Folder of Folders...')

    st = time.time()

    existing_merged = None
    existing_embedded_srcs = []
    if embedding_config_hash:
      print('Checking for existing embeddings from Folder of Folders...')

      base_path = epath.Path(base_dir)
      embedding_folder = (
          base_path / f'{embedding_file_prefix}{embedding_config_hash}'
      )

      if embedding_folder.exists() and any(embedding_folder.iterdir()):
        existing_merged = cls.from_tfrecords(
            base_dir, embedding_folder.as_posix(), time_pooling, exclude_classes
        )
        existing_embedded_srcs = existing_merged.data['filename']

      print(f'Found {len(existing_embedded_srcs)} existing embeddings.')

    print('Checking for new sources to embed from Folder of Folders...')

    labels, merged = embed_dataset(
        base_dir=base_dir,
        embedding_model=embedding_model,
        time_pooling=time_pooling,
        exclude_classes=exclude_classes,
        load_audio=load_audio,
        target_sample_rate=target_sample_rate,
        audio_file_pattern=audio_file_pattern,
        excluded_files=existing_embedded_srcs,
        embedding_file_prefix=embedding_file_prefix,
        pad_type=pad_type,
        max_workers=max_workers,
    )

    if not merged and existing_merged is None:
      raise ValueError('No embeddings or raw audio files found.')

    if not merged and existing_merged is not None:
      print('\nUsing existing embeddings for all audio source files.')
      return existing_merged

    elapsed = time.time() - st
    print(f'\n...embedded dataset in {elapsed:5.2f}s...')
    data = merged
    embedding_dim = merged['embeddings'].shape[-1]

    labels = tuple(labels)
    num_classes = len(labels)
    print(f'    found {num_classes} classes.')
    class_counts = collections.defaultdict(int)
    for cl, cl_str in zip(merged['label'], merged['label_str']):
      class_counts[(cl, cl_str)] += 1
    for (cl, cl_str), count in sorted(class_counts.items()):
      print(f'    class {cl_str} / {cl} : {count}')
    new_merged = cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

    if cache_embeddings:
      if not embedding_config_hash:
        raise ValueError(
            'Embedding config hash must be specified when caching embeddings.'
        )

      new_merged.write_embeddings_to_tf_records(
          base_dir,
          embedding_config_hash,
          embedding_file_prefix,
          tf_record_shards,
      )

    if existing_merged:
      return cls.from_merged_datasets([new_merged, existing_merged])

    return new_merged

  @classmethod
  def from_tfrecords(
      cls,
      base_dir: str,
      embeddings_path: str,
      time_pooling: str,
      exclude_classes: Sequence[str] = (),
  ) -> 'MergedDataset':
    """Generating MergedDataset via reading existing embeddings.

    Note: this assumes the embeddings were run with folder_of_folders
    with file_id_depth=1 in the embeddings export. This classmethod can/will be
    updated for allowing a few options for specifying labels.

    Args:
      base_dir: Base directory (folder of folders of original audio)
      embeddings_path: Location of the existing embeddings.
      time_pooling: Method of time pooling.
      exclude_classes: List of classes to exclude.

    Returns:
      MergedDataset
    """
    labels, merged = read_embedded_dataset(
        base_dir=base_dir,
        embeddings_path=embeddings_path,
        time_pooling=time_pooling,
        exclude_classes=exclude_classes,
    )
    data = merged
    embedding_dim = merged['embeddings'].shape[-1]

    labels = tuple(labels)
    num_classes = len(labels)
    print(f'    found {num_classes} classes.')
    class_counts = collections.defaultdict(int)
    for cl, cl_str in zip(merged['label'], merged['label_str']):
      class_counts[(cl, cl_str)] += 1
    for (cl, cl_str), count in sorted(class_counts.items()):
      print(f'    class {cl_str} / {cl} : {count}')
    return cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

  @classmethod
  def from_merged_datasets(
      cls, merged_datasets: Sequence['MergedDataset']
  ) -> 'MergedDataset':
    """Generating MergedDataset from a sequence of MergedDatasets.

    This assumes that the given merged datasets are compatible, i.e. they were
    generated with the same options and embedding configurations.

    Args:
      merged_datasets: Sequence of compatible MergedDatasets.

    Returns:
      MergedDataset
    """

    embedding_dim = merged_datasets[0].embedding_dim
    num_classes = merged_datasets[0].num_classes
    labels = merged_datasets[0].labels
    data = {}

    for merged_dataset in merged_datasets[1:]:
      # TODO: Improve compatibility checking to use config hashes.
      if (
          embedding_dim != merged_dataset.embedding_dim
          or num_classes != merged_dataset.num_classes
          or labels != merged_dataset.labels
      ):
        raise ValueError('Given merged datasets are not compatible.')

    for key in merged_datasets[0].data.keys():
      data_arrays = [merged_data.data[key] for merged_data in merged_datasets]
      data[key] = np.concatenate(data_arrays)

    return cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

  def embeddings_to_tf_examples(self) -> Sequence[tf.train.Example]:
    """Return a dictionary of embedding tf.Examples keyed by label_str."""
    examples = []
    embeddings = self.data['embeddings']
    filename = self.data['filename']

    for embedding, filename in zip(embeddings, filename):
      examples.append(
          tf_examples.model_outputs_to_tf_example(
              model_outputs=interface.InferenceOutputs(embedding),
              file_id=filename,
              audio=np.empty(1),
              timestamp_offset_s=0,
              write_embeddings=True,
              write_logits=False,
              write_separated_audio=False,
              write_raw_audio=False,
          )
      )

    return examples

  def write_embeddings_to_tf_records(
      self,
      base_dir: str,
      embedding_config_hash: str,
      embedding_file_prefix: str = 'embeddings-',
      tf_record_shards: int = 1,
  ) -> None:
    """Materialize MergedDataset embeddings as TF records to folder-of-folders.

    Args:
      base_dir: Base directory where either folder-of-folders of audio or
        tfrecord embeddings are stored.
      embedding_config_hash: String hash of the embedding config to identify an
        existing embeddings folder. This will be appended to the
        embedding_file_prefix, e.g. 'embeddings-1234'.
      embedding_file_prefix: Prefix for existing materialized embedding files.
        Embeddings with a matching hash will be re-used to avoid reprocessing,
        and embeddings with a non-matching hash will be ignored.
      tf_record_shards: Number of files to materialize if writing embeddings to
        TF records.
    """
    embedding_examples = self.embeddings_to_tf_examples()
    output_dir = (
        epath.Path(base_dir) / f'{embedding_file_prefix}{embedding_config_hash}'
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir=output_dir.as_posix(), num_files=tf_record_shards
    ) as file_writer:
      for example in embedding_examples:
        file_writer.write(example.SerializeToString())
      file_writer.flush()

  def create_random_train_test_split(
      self,
      train_ratio: float | None,
      train_examples_per_class: int | None,
      seed: int,
      exclude_classes: Sequence[int] = (),
      exclude_eval_classes: Sequence[int] = (),
  ):
    """Generate a train/test split with a target number of train examples."""
    if train_ratio is None and train_examples_per_class is None:
      raise ValueError(
          'Must specify one of train_ratio and examples_per_class.'
      )
    elif train_ratio is not None and train_examples_per_class is not None:
      raise ValueError(
          'Must specify only one of train_ratio and examples_per_class.'
      )

    # Use a seeded shuffle to get a random ordering of the data.
    locs = list(range(self.data['label'].shape[0]))
    np.random.seed(seed)
    np.random.shuffle(locs)

    classes = set(self.data['label'])
    class_counts = {cl: np.sum(self.data['label'] == cl) for cl in classes}
    if train_examples_per_class is not None:
      class_limits = {cl: train_examples_per_class for cl in classes}
    else:
      class_limits = {cl: train_ratio * class_counts[cl] for cl in classes}

    classes = set(self.data['label'])
    class_locs = {cl: [] for cl in classes}
    train_locs = []
    test_locs = []
    for loc in locs:
      cl = self.data['label'][loc]
      if cl in exclude_classes:
        continue
      if len(class_locs[cl]) < class_limits[cl]:
        class_locs[cl].append(loc)
        train_locs.append(loc)
      elif cl not in exclude_eval_classes:
        test_locs.append(loc)
    train_locs = np.array(train_locs)
    test_locs = np.array(test_locs)
    return train_locs, test_locs, class_locs


def pool_time_axis(embeddings, pool_method, axis=1):
  """Apply pooling over the specified axis."""
  if pool_method == 'mean':
    if embeddings.shape[axis] == 0:
      return embeddings.sum(axis=axis)
    return embeddings.mean(axis=axis)
  elif pool_method == 'max':
    return embeddings.max(axis=axis)
  elif pool_method == 'mid':
    t = embeddings.shape[axis] // 2
    return embeddings[:, t]
  elif pool_method == 'flatten':
    if len(embeddings.shape) != 3 and axis != 1:
      raise ValueError(
          'Can only flatten time for embeddings with shape [B, T, D].'
      )
    depth = embeddings.shape[-1]
    time_steps = embeddings.shape[1]
    return embeddings.reshape([embeddings.shape[0], time_steps * depth])
  raise ValueError('Unrecognized reduction method.')


def _pad_audio(
    audio: np.ndarray, target_length: int, pad_type: str = 'zeros'
) -> np.ndarray:
  """Pad audio to target_length."""
  if len(audio.shape) > 1:
    raise ValueError('audio should be a flat array.')
  if audio.shape[0] >= target_length:
    return audio
  if pad_type == 'zeros':
    pad_amount = target_length - audio.shape[0]
    front = pad_amount // 2
    back = pad_amount - front
    return np.pad(audio, [(front, back)], 'constant')
  elif pad_type == 'repeat':
    # repeat audio until longer than target_length.
    num_repeats = target_length // audio.shape[0] + 1
    repeated_audio = np.repeat(audio, num_repeats, axis=0)
    start = repeated_audio.shape[0] - target_length // 2
    padded = repeated_audio[start : start + target_length]
    return padded
  raise ValueError('Unrecognized padding method.')


# TODO: add alternative labeling strategies as options
def labels_from_folder_of_folders(
    base_dir: str,
    exclude_classes: Sequence[str] = (),
    embedding_file_prefix: str = 'embeddings-',
) -> Sequence[str]:
  """Returns the labels from the given folder of folders.

  Args:
    base_dir: Folder of folders directory containing audio or embedded data.
    exclude_classes: Classes to skip.
    embedding_file_prefix: Folders containing existing embeddings that will be
      ignored when determining labels.
  """
  base_dir = epath.Path(base_dir)
  sub_dirs = sorted([p.name for p in base_dir.glob('*') if p.is_dir()])
  if not sub_dirs:
    raise ValueError(
        'No subfolders found in base directory. Audio will be '
        'matched as "base_dir/*/*.wav", with the subfolders '
        'indicating class names.'
    )

  labels = []
  for d in sub_dirs:
    if d in exclude_classes:
      continue
    if d.startswith(embedding_file_prefix):
      continue
    labels.append(d)

  return labels


def embed_dataset(
    base_dir: str,
    embedding_model: interface.EmbeddingModel,
    time_pooling: str,
    exclude_classes: Sequence[str] = (),
    load_audio: bool = False,
    target_sample_rate: int = -1,
    audio_file_pattern: str = '*',
    excluded_files: Sequence[str] = (),
    embedding_file_prefix: str = 'embeddings-',
    pad_type: str = 'zeros',
    max_workers: int = 5,
) -> Tuple[Sequence[str], Dict[str, np.ndarray]]:
  """Add embeddings to an eval dataset.

  Embed a dataset, creating an in-memory copy of all data with embeddings added.
  The base_dir should contain folders corresponding to classes, and each
  sub-folder should contina audio files for the respective class.

  Note that any audio files in the base_dir directly will be ignored.

  Args:
    base_dir: Directory contianing audio data.
    embedding_model: Model for computing audio embeddings.
    time_pooling: Key for time pooling strategy.
    exclude_classes: Classes to skip.
    load_audio: Whether to load audio into memory.
    target_sample_rate: Resample loaded audio to this sample rate. If -1, loads
      raw audio with no resampling. If -2, uses the embedding_model sample rate.
    audio_file_pattern: The glob pattern to use for finding audio files within
      the sub-folders.
    excluded_files: These files will be ignored, the paths are assumed to be
      relative to the base_dir.
    embedding_file_prefix: Prefix for existing embedding files, matching files
      will be ignored.
    pad_type: Padding style for short audio.
    max_workers: Number of threads to use for loading audio.

  Returns:
    Ordered labels and a Dict contianing the entire embedded dataset.
  """
  labels = labels_from_folder_of_folders(base_dir, exclude_classes)
  base_dir = epath.Path(base_dir)

  if hasattr(embedding_model, 'window_size_s'):
    window_size = int(
        embedding_model.window_size_s * embedding_model.sample_rate
    )
  else:
    window_size = -1

  if target_sample_rate == -2:
    target_sample_rate = embedding_model.sample_rate

  merged = collections.defaultdict(list)
  for label_idx, label in enumerate(labels):
    label_hot = np.zeros([len(labels)], np.int32)
    label_hot[label_idx] = 1

    # Get filepaths excluding embedding files
    filepaths = [
        fp
        for fp in (base_dir / label).glob(audio_file_pattern)
        if not fp.name.startswith(embedding_file_prefix)
    ]

    if not filepaths:
      raise ValueError(
          'No files matching {} were found in directory {}'.format(
              audio_file_pattern, base_dir / label
          )
      )

    filepaths = [
        fp.as_posix()
        for fp in filepaths
        if fp.relative_to(base_dir).as_posix() not in excluded_files
    ]

    audio_loader = lambda fp, offset: _pad_audio(
        np.asarray(audio_utils.load_audio(fp, target_sample_rate)),
        window_size,
        pad_type,
    )
    audio_iterator = audio_utils.multi_load_audio_window(
        audio_loader=audio_loader,
        filepaths=filepaths,
        offsets=None,
        max_workers=max_workers,
        buffer_size=64,
    )

    for audio in tqdm.tqdm(audio_iterator):
      outputs = embedding_model.embed(audio)
      if outputs.embeddings is None:
        raise ValueError('Embedding model did not produce any embeddings!')
      # If the audio was separated then the raw audio is in the first channel.
      # Embedding shape is either [B, F, C, D] or [F, C, D] so channel is
      # always -2.
      channel_pooling = (
          'squeeze' if outputs.embeddings.shape[-2] == 1 else 'first'
      )
      embeds = outputs.pooled_embeddings(time_pooling, channel_pooling)
      merged['embeddings'].append(embeds)
      if load_audio:
        merged['audio'].append(audio)

    for fp in filepaths:
      filename = epath.Path(fp).name
      merged['filename'].append(f'{label}/{filename}')
      merged['label'].append(label_idx)
      merged['label_str'].append(label)
      merged['label_hot'].append(label_hot)

  outputs = {}
  for k in merged.keys():
    outputs[k] = np.stack(merged[k])
  return labels, outputs


def read_embedded_dataset(
    base_dir: str,
    embeddings_path: str,
    time_pooling: str,
    exclude_classes: Sequence[str] = (),
    tensor_dtype: str = 'float32',
):
  """Read pre-saved embeddings to memory from storage.

  This function reads a set of embeddings that has already been generated
  to load as a MergedDataset via from_tfrecords(). The embeddings could be saved
  in one folder or be contained in multiple subdirectories. This function
  produces the same output as embed_dataset(), except (for now) we don't allow
  for the optional loading of the audio (.wav files). However, for labeled data,
  we still require the base directory containing the folder-of-folders with the
  audio data to produce the labels. If there are no subfolders, no labels will
  be created.

  Args:
    base_dir: Base directory where audio may be stored in a subdirectories,
      where the folder represents the label
    embeddings_path: Location of the existing embeddings as TFRecordDataset.
    time_pooling: Method of time pooling.
    exclude_classes: List of classes to exclude.
    tensor_dtype: Tensor dtype used in the embeddings tfrecords.

  Returns:
    Ordered labels and a Dict contianing the entire embedded dataset.
  """

  output_dir = epath.Path(embeddings_path)
  fns = [fn for fn in output_dir.glob('embeddings-*')]
  ds = tf.data.TFRecordDataset(fns)
  parser = tf_examples.get_example_parser(tensor_dtype=tensor_dtype)
  ds = ds.map(parser)

  labels = labels_from_folder_of_folders(base_dir, exclude_classes)

  merged = collections.defaultdict(list)
  label_dict = collections.defaultdict(dict)

  for label_idx, label in enumerate(labels):
    label_hot = np.zeros([len(labels)], np.int32)
    label_hot[label_idx] = 1

    label_dict[label]['label_hot'] = label_hot
    label_dict[label]['label_idx'] = label_idx
    label_dict[label]['label_str'] = label

  for ex in ds.as_numpy_iterator():
    # Embedding has already been pooled into single dim.
    if len(ex['embedding'].shape) == 1:
      outputs = ex['embedding']
    else:
      outputs = interface.pool_axis(ex['embedding'], -3, time_pooling)
      if ex['embedding'].shape[-2] == 1:
        channel_pooling = 'squeeze'
      else:
        channel_pooling = 'first'
      outputs = interface.pool_axis(outputs, -2, channel_pooling)

    merged['embeddings'].append(outputs)
    merged['filename'].append(ex['filename'].decode())
    file_label = ex['filename'].decode().split('/')[0]
    merged['label'].append(label_dict[file_label]['label_idx'])
    merged['label_str'].append(label_dict[file_label]['label_str'])
    merged['label_hot'].append(label_dict[file_label]['label_hot'])

  outputs = {}
  for k in merged.keys():
    outputs[k] = np.stack(merged[k])
  return labels, outputs
