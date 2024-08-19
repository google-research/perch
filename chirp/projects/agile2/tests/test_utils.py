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

"""Utilities for testing agile modeling functionality."""

import os

from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from ml_collections import config_dict
import numpy as np
from scipy.io import wavfile


CLASS_LABELS = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta')


def make_wav_files(
    base_path, classes, filenames, file_len_s=1.0, sample_rate_hz=16000
):
  """Create a pile of wav files in a directory structure."""
  rng = np.random.default_rng(seed=42)
  for subdir in classes:
    subdir_path = os.path.join(base_path, subdir)
    os.mkdir(subdir_path)
    for filename in filenames:
      with open(
          os.path.join(subdir_path, f'{filename}_{subdir}.wav'), 'wb'
      ) as f:
        noise = rng.normal(scale=0.2, size=int(file_len_s * sample_rate_hz))
        wavfile.write(f, sample_rate_hz, noise)
  audio_glob = os.path.join(base_path, '*/*.wav')
  return audio_glob


def make_db(
    path: str,
    db_type: str,
    num_embeddings: int,
    rng: np.random.Generator,
    embedding_dim: int = 128,
) -> interface.GraphSearchDBInterface:
  """Create a test DB of the specified type."""
  if db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(
        embedding_dim=embedding_dim,
        max_size=5000,
        degree_bound=256,
    )
  elif db_type == 'sqlite':
    # TODO(tomdenton): use tempfile.
    db = sqlite_impl.SQLiteGraphSearchDB.create(
        db_path=os.path.join(path, 'db.sqlite'),
        embedding_dim=embedding_dim,
    )
  else:
    raise ValueError(f'Unknown db type: {db_type}')
  db.setup()
  # Insert a few embeddings...
  graph_utils.insert_random_embeddings(db, embedding_dim, num_embeddings, rng)

  config = config_dict.ConfigDict()
  config.embedding_dim = embedding_dim
  db.insert_metadata('db_config', config)
  db.commit()
  return db


def add_random_labels(
    db: interface.GraphSearchDBInterface,
    rng: np.random.Generator,
    unlabeled_prob: float = 0.5,
    positive_label_prob: float = 0.5,
    provenance: str = 'test',
):
  """Insert random labels for a subset of embeddings."""
  for idx in db.get_embedding_ids():
    if rng.random() < unlabeled_prob:
      continue
    if rng.random() < positive_label_prob:
      label_type = interface.LabelType.POSITIVE
    else:
      label_type = interface.LabelType.NEGATIVE
    label = interface.Label(
        embedding_id=idx,
        label=str(rng.choice(CLASS_LABELS)),
        type=label_type,
        provenance=provenance,
    )
    db.insert_label(label)
  db.commit()
