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
import time
from typing import Dict, Sequence, Tuple

from chirp import audio_utils
from chirp.inference import interface
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
      load_audio: bool = True,
      target_sample_rate: int = -2,
      audio_file_pattern: str = '*',
  ) -> 'MergedDataset':
    """Generating MergedDataset via folder-of-folders method.

    Args:
      base_dir: Base directory where either folder-of-folders of audio or
        tfrecord embeddings are stored.
      embedding_model: EmbeddingModel used to produce embeddings.
      time_pooling: Key for time pooling strategy.
      exclude_classes: Classes to skip.
      load_audio: Whether to load audio into memory.
      target_sample_rate: Resample loaded audio to this sample rate. If -1,
        loads raw audio with no resampling. If -2, uses the embedding_model
        sample rate.
      audio_file_pattern: The glob pattern to use for finding audio files within
        the sub-folders.

    Returns:
      MergedDataset
    """
    print('Embedding from Folder of Folders...')

    st = time.time()
    labels, merged = embed_dataset(
        base_dir=base_dir,
        embedding_model=embedding_model,
        time_pooling=time_pooling,
        exclude_classes=exclude_classes,
        load_audio=load_audio,
        target_sample_rate=target_sample_rate,
        audio_file_pattern=audio_file_pattern,
    )
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
    return cls(
        data=data,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        labels=labels,
    )

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

  def create_keras_dataset(
      self, locs: Sequence[int], is_train: bool, batch_size: int
  ) -> tf.data.Dataset:
    """Create a keras-friendly tf.data.Dataset from the in-memory dataset."""

    def _data_gen():
      for loc in locs:
        yield (
            self.data['embeddings'][loc],
            tf.one_hot(self.data['label'][loc], self.num_classes),
        )

    ds = tf.data.Dataset.from_generator(
        _data_gen,
        output_types=(tf.float32, tf.int64),
        output_shapes=(self.embedding_dim, self.num_classes),
    )
    if is_train:
      ds = ds.shuffle(1024)
    ds = ds.batch(batch_size)
    return ds


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


def _pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
  if len(audio.shape) > 1:
    raise ValueError('audio should be a flat array.')
  if audio.shape[0] > target_length:
    return audio
  pad_amount = target_length - audio.shape[0]
  front = pad_amount // 2
  back = pad_amount - front
  return np.pad(audio, [(front, back)], 'constant')


def embed_dataset(
    base_dir: str,
    embedding_model: interface.EmbeddingModel,
    time_pooling: str,
    exclude_classes: Sequence[str] = (),
    load_audio: bool = True,
    target_sample_rate: int = -1,
    audio_file_pattern: str = '*',
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

  Returns:
    Ordered labels and a Dict contianing the entire embedded dataset.
  """
  base_dir = epath.Path(base_dir)
  labels = sorted([p.name for p in base_dir.glob('*') if p.is_dir()])
  if not labels:
    raise ValueError(
        'No subfolders found in base directory. Audio will be '
        'matched as "base_dir/*/*.wav", with the subfolders '
        'indicating class names.'
    )
  labels = [label for label in labels if label not in exclude_classes]

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

    filepaths = [
        fp.as_posix() for fp in (base_dir / label).glob(audio_file_pattern)
    ]

    if not filepaths:
      raise ValueError(
          'No files matching {} were found in directory {}'.format(
              audio_file_pattern, base_dir / label
          )
      )

    audio_iterator = audio_utils.multi_load_audio_window(
        filepaths, None, target_sample_rate, -1
    )

    for fp, audio in tqdm.tqdm(
        zip(filepaths, audio_iterator), total=len(filepaths)
    ):
      audio_size = audio.shape[0]
      if window_size > audio_size:
        audio = _pad_audio(audio, window_size)
      audio = audio.astype(np.float32)
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

      filename = epath.Path(fp).name
      merged['filename'].append(f'{label}/{filename}')
      if load_audio:
        merged['audio'].append(audio)
      merged['label'].append(label_idx)
      merged['label_str'].append(label)
      merged['label_hot'].append(label_hot)

  if load_audio:
    # pad audio to ensure all the same length.
    target_audio_len = np.max([a.shape[0] for a in merged['audio']])
    merged['audio'] = [_pad_audio(a, target_audio_len) for a in merged['audio']]

  outputs = {}
  for k in merged.keys():
    outputs[k] = np.stack(merged[k])
  return labels, outputs
