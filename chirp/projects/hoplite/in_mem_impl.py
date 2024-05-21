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

"""In-Memory implementation of GraphSearchDBInterface for testing."""

import dataclasses
from typing import Any, Sequence

from chirp.projects.hoplite import interface
import numpy as np


@dataclasses.dataclass
class InMemoryGraphSearchDB(interface.GraphSearchDBInterface):
  """SQLite implementation of graph search database."""

  embeddings: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
  edges: dict[int, tuple[int, ...]] = dataclasses.field(default_factory=dict)
  metadata: dict[int, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def create(cls, **kwargs):
    """Connect to and, if needed, initialize the database."""
    return cls(**kwargs)

  def setup(self):
    """Initialize an empty database."""
    self.embeddings = {}
    self.edges = {}
    self.metadata = {}

  def count_embeddings(self) -> int:
    """Return a count of all embeddings in the database."""
    return len(self.embeddings)

  def count_edges(self) -> int:
    """Return a count of all embeddings in the database."""
    return sum(len(edges) for edges in self.edges.values())

  def insert_edge(self, x_id: int, y_id: int) -> None:
    """Add an edge between two embeddings."""
    self.edges[x_id] = self.edges.get(x_id, tuple()) + (y_id,)

  def delete_edge(self, x_id, y_id) -> None:
    """Delete an edge between two embeddings."""
    x_edges = self.edges.get(x_id, tuple())
    self.edges[x_id] = tuple(z for z in x_edges if z != y_id)

  def delete_edges(self, x_id) -> None:
    """Delete all edges originating from x_id."""
    self.edges.pop(x_id)

  def drop_all_edges(self) -> None:
    """Delete all edges in the database."""
    self.edges = {}

  def insert_embedding(self, embedding, **metadata) -> int:
    """Add an embedding with metadata to the database."""
    idx = len(self.embeddings) + 1
    self.embeddings[idx] = embedding
    self.metadata[idx] = metadata
    return idx

  def get_embedding(self, embedding_id: int) -> np.ndarray:
    """Retrieve an embedding from the database."""
    return self.embeddings[embedding_id]

  def get_embeddings(self, embedding_ids: Sequence[int]) -> np.ndarray:
    return np.array(
        tuple(self.embeddings[embedding_id] for embedding_id in embedding_ids)
    )

  def get_embedding_ids(self) -> tuple[int, ...]:
    """Get all embedding IDs in the database."""
    return tuple(self.embeddings.keys())

  def get_metadata(self, embedding_id: int) -> dict[str, Any]:
    return self.metadata[embedding_id]

  def get_edges(self, embedding_id: int) -> tuple[int, ...]:
    return self.edges.get(embedding_id, tuple())

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    pass
