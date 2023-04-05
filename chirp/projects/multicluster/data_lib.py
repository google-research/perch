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
import os
import time
from typing import Dict, Optional, Sequence, Tuple

from chirp.inference import interface
from etils import epath
import librosa
import numpy as np
import tensorflow as tf


@dataclasses.dataclass
class MergedDataset:
  """In-memory dataset of labeled audio with embeddings."""

  base_dir: str
  embedding_model: interface.EmbeddingModel
  num_splits: int = 5
  time_pooling: str = 'mean'
  exclude_classes: Sequence[str] = ()
  exclude_eval_classes: Sequence[str] = ()
  negative_label: str = 'unknown'

  # The following are populated automatically.
  data: Optional[Dict[str, np.ndarray]] = None
  num_classes: Optional[int] = None
  data_sample_rate: Optional[int] = None
  embedding_dim: Optional[int] = None
  label_lookup: Optional[tf.lookup.StaticHashTable] = None

  def __post_init__(self):
    wavs_dataset, label_lookup = dataset_from_labeled_wav_dirs(
        self.base_dir, self.num_splits, self.exclude_classes
    )
    self.label_lookup = label_lookup
    for ex in wavs_dataset.as_numpy_iterator():
      self.data_sample_rate = ex['sample_rate']
      break

    st = time.time()
    merged = embed_dataset(
        self.embedding_model,
        wavs_dataset,
        self.data_sample_rate,
        self.time_pooling,
        self.exclude_classes,
    )
    elapsed = time.time() - st
    print(f'\n...embedded dataset in {elapsed:5.2f}s...')
    self.data = merged
    self.embedding_dim = merged['embeddings'].shape[-1]

    self.num_classes = self.label_lookup.size()
    if hasattr(self.num_classes, 'numpy'):
      self.num_classes = self.num_classes.numpy()
    print(f'    found {self.num_classes} classes.')
    class_counts = collections.defaultdict(int)
    for cl, cl_str in zip(merged['label'], merged['label_str']):
      class_counts[(cl, cl_str)] += 1
    for (cl, cl_str), count in sorted(class_counts.items()):
      print(f'    class {cl_str} / {cl} : {count}')

  def create_random_train_test_split(
      self,
      examples_per_class: int,
      seed: int,
      exclude_classes: Sequence[int] = (),
      exclude_eval_classes: Sequence[int] = (),
  ):
    """Generate a train/test split with a target number of train examples."""
    # Use a seeded shuffle to get a random ordering of the data.
    locs = list(range(self.data['label'].shape[0]))
    np.random.seed(seed)
    np.random.shuffle(locs)

    classes = set(self.data['label'])
    class_locs = {cl: [] for cl in classes}
    train_locs = []
    test_locs = []
    for loc in locs:
      cl = self.data['label'][loc]
      if cl in exclude_classes:
        continue
      if len(class_locs[cl]) < examples_per_class:
        class_locs[cl].append(loc)
        train_locs.append(loc)
      elif cl not in exclude_eval_classes:
        test_locs.append(loc)
    train_locs = np.array(train_locs)
    test_locs = np.array(test_locs)
    return train_locs, test_locs, class_locs

  def create_keras_dataset(
      self, locs: np.ndarray, is_train: bool, batch_size: int
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


def progress_dot(i: int, print_mod: int = 10, break_mod: Optional[int] = None):
  """Print a dot, with occasional line breaks."""
  if break_mod is None:
    break_mod = 25 * print_mod
  if (i + 1) % break_mod == 0:
    print('.')
  elif (i + 1) % print_mod == 0 or print_mod <= 1:
    print('.', end='')


def dataset_from_labeled_wav_dirs(
    base_dir: str, num_splits: int, exclude_classes: Sequence[str] = ()
) -> Tuple[tf.data.Dataset, tf.lookup.StaticHashTable]:
  """Create a dataset from wavs organized into label directories."""
  wavs_glob = os.path.join(base_dir, '*/*.wav')

  # Count the number of labels.
  p = epath.Path(base_dir)
  labels = sorted(
      [
          lbl.stem
          for lbl in p.iterdir()
          if lbl.is_dir() and lbl.stem not in exclude_classes
      ]
  )
  label_lookup = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(labels, range(len(labels))),
      len(labels),
  )

  def _read_wav(filename):
    """TF function for creating wav features with a filepath label."""
    file_contents = tf.io.read_file(filename)
    try:
      wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    except tf.errors.OpError as e:
      raise ValueError(f'Failed to decode wav file ({filename})') from e
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    label = tf.strings.split(filename, '/')[-2]
    split = tf.strings.to_hash_bucket(filename, num_splits)
    features = {
        'filename': filename,
        'split': split,
        'audio': wav,
        'label': label_lookup.lookup(label),
        'label_str': label,
        'label_hot': tf.one_hot(label_lookup.lookup(label), len(labels)),
        'sample_rate': sample_rate,
    }
    return features

  ds = tf.data.Dataset.list_files(wavs_glob).map(_read_wav)
  return ds, label_lookup


def pool_time_axis(embeddings, pool_method, axis=1):
  """Apply pooling over the specified axis."""
  if pool_method == 'mean':
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
    embedding_model: interface.EmbeddingModel,
    dataset: tf.data.Dataset,
    data_sample_rate: int,
    time_pooling: str,
    exclude_classes: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
  """Add embeddings to an eval dataset.

  Embed a dataset, creating an in-memory copy of all data with embeddings added.

  Args:
    embedding_model: Inference model.
    dataset: TF Dataset of unbatched audio examples.
    data_sample_rate: Sample rate of dataset audio.
    time_pooling: Key for time pooling strategy.
    exclude_classes: Classes to skip.

  Returns:
    Dict contianing the entire embedded dataset.
  """
  merged = collections.defaultdict(list)
  exclude_classes = set(exclude_classes)
  for i, ex in enumerate(dataset.as_numpy_iterator()):
    if ex['label_str'] in exclude_classes:
      continue
    if data_sample_rate > 0 and data_sample_rate != embedding_model.sample_rate:
      ex['audio'] = librosa.resample(
          ex['audio'],
          data_sample_rate,
          embedding_model.sample_rate,
          res_type='polyphase',
      )

    audio_size = ex['audio'].shape[0]
    if hasattr(embedding_model, 'window_size_s'):
      window_size = int(
          embedding_model.window_size_s * embedding_model.sample_rate
      )
      if window_size > audio_size:
        ex['audio'] = _pad_audio(ex['audio'], window_size)

    outputs = embedding_model.embed(ex['audio'])
    if outputs.embeddings is not None:
      embeds = outputs.pooled_embeddings(time_pooling, 'squeeze')
      ex['embeddings'] = embeds
    if outputs.separated_audio is not None:
      ex['separated_audio'] = outputs.separated_audio

    progress_dot(i)

    for k in ex.keys():
      merged[k].append(ex[k])

  # pad audio to ensure all the same length.
  target_audio_len = np.max([a.shape[0] for a in merged['audio']])
  merged['audio'] = [_pad_audio(a, target_audio_len) for a in merged['audio']]

  outputs = {}
  for k in merged.keys():
    if k == 'embeddings':
      print([x.shape for x in merged[k]])
    outputs[k] = np.stack(merged[k])

  # Check that all keys have the same batch dimension.
  batch_size = outputs['embeddings'].shape[0]
  for k in outputs:
    if outputs[k].shape[0] != batch_size:
      mismatched = outputs[k].shape[0]
      raise ValueError(
          f'Size mismatch between embeddings ({batch_size}) '
          f'and {k} ({mismatched})'
      )

  return outputs
