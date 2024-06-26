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

"""Database configuration and constructor."""

import dataclasses
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from ml_collections import config_dict


@dataclasses.dataclass
class DBConfig(interface.EmbeddingMetadata):
  """Configuration for embedding database.

  Attributes:
    db_key: Key for the database implementation to use.
    db_config: Configuration for the database implementation.
  """

  db_key: str
  db_config: config_dict.ConfigDict

  def load_db(self) -> interface.GraphSearchDBInterface:
    """Load the database from the specified path."""
    if self.db_key == 'sqlite':
      return sqlite_impl.SQLiteGraphSearchDB.create(**self.db_config)
    elif self.db_key == 'in_mem':
      return in_mem_impl.InMemoryGraphSearchDB.create(**self.db_config)
    else:
      raise ValueError(f'Unknown db_key: {self.db_key}')
