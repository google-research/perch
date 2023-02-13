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
from typing import Dict, Optional

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
  num_splits: int

  # The following are populated automatically.
  data: Optional[Dict[str, np.ndarray]] = None
  num_classes: Optional[int] = None
  data_sample_rate: Optional[int] = None
  embedding_dim: Optional[int] = None

  def __post_init__(self):
    wavs_dataset = dataset_from_labeled_wav_dirs(self.base_dir, self.num_splits)
    for ex in wavs_dataset.as_numpy_iterator():
      self.data_sample_rate = ex['sample_rate']
      break

    st = time.time()
    merged = embed_dataset(
        self.embedding_model, wavs_dataset, self.data_sample_rate
    )
    elapsed = time.time() - st
    print(f'\n...embedded dataset in {elapsed:5.2f}s...')
    self.data = merged
    self.embedding_dim = merged['embeddings'].shape[-1]

    self.num_classes = len(set(merged['label']))
    print(f'    found {self.num_classes} classes.')
    class_counts = collections.defaultdict(int)
    for cl in merged['label']:
      class_counts[cl] += 1
    for cl, count in class_counts.items():
      print(f'    class {cl} : {count}')

  def create_random_train_test_split(self, examples_per_class: int, seed: int):
    """Generate a random train/test split."""
    locs = list(range(self.data['label'].shape[0]))
    np.random.seed(seed)
    np.random.shuffle(locs)

    classes = set(self.data['label'])
    class_locs = {cl: [] for cl in classes}
    train_locs = []
    test_locs = []
    for loc in locs:
      cl = self.data['label'][loc]
      if len(class_locs[cl]) < examples_per_class:
        class_locs[cl].append(loc)
        train_locs.append(loc)
      else:
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
    base_dir: str, num_splits: int
) -> tf.data.Dataset:
  """Create a dataset from wavs organized into label directories."""
  wavs_glob = os.path.join(base_dir, '*/*.wav')

  # Count the number of labels.
  p = epath.Path(base_dir)
  labels = [lbl.stem for lbl in p.iterdir() if lbl.is_dir()]
  label_lookup = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(labels, range(len(labels))),
      len(labels),
  )

  def _read_wav(filename):
    """TF function for creating wav features with a filepath label."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
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
        'sample_rate': sample_rate,
    }
    return features

  ds = tf.data.Dataset.list_files(wavs_glob).map(_read_wav)
  return ds


def embed_dataset(
    embedding_model: interface.EmbeddingModel,
    dataset: tf.data.Dataset,
    data_sample_rate: int,
) -> Dict[str, np.ndarray]:
  """Add embeddings to an eval dataset.

  Embed a dataset, creating an in-memory copy of all data with embeddings added.

  Args:
    embedding_model: Inference model.
    dataset: TF Dataset of unbatched audio examples.
    data_sample_rate: Sample rate of dataset audio.

  Returns:
    Dict contianing the entire embedded dataset.
  """
  merged = {}
  for i, ex in enumerate(dataset.as_numpy_iterator()):
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
        pad_amount = window_size - audio_size
        front = pad_amount // 2
        back = pad_amount - front + pad_amount % 2
        ex['audio'] = np.pad(ex['audio'], [(front, back)], 'constant')

    outputs = embedding_model.embed(ex['audio'])
    if outputs.logits is not None:
      k = list(outputs.logits.keys())[0]
      logits = outputs.logits[k].mean(axis=1).squeeze()
      ex['logits'] = logits
    if outputs.embeddings is not None:
      embeds = outputs.embeddings.mean(axis=1).squeeze()
      ex['embeddings'] = embeds
    if outputs.separated_audio is not None:
      ex['separated_audio'] = outputs.separated_audio

    progress_dot(i)

    if not merged:
      merged = {k: [ex[k]] for k in ex.keys()}
    else:
      for k in ex.keys():
        merged[k].append(ex[k])
  merged = {k: np.stack(merged[k]) for k in merged.keys()}
  return merged
