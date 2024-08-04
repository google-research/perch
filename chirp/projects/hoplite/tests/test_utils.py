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

"""Utility functions for testing."""

import os
from chirp.projects.hoplite import graph_utils
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from ml_collections import config_dict
import numpy as np


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
        max_size=2000,
        degree_bound=1000,
    )
  elif db_type == 'sqlite':
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


def clone_embeddings(
    source_db: interface.GraphSearchDBInterface,
    target_db: interface.GraphSearchDBInterface,
):
  """Copy all embeddings to target_db and provide an id mapping."""
  id_mapping = {}
  for source_id in source_db.get_embedding_ids():
    id_mapping[source_id] = target_db.insert_embedding(
        source_db.get_embedding(source_id),
        source_db.get_embedding_source(source_id),
    )
  return id_mapping
