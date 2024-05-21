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

"""Interface for a searchable embeddings database."""

from collections.abc import Sequence
from typing import Any
import numpy as np


class GraphSearchDBInterface:
  """Interface for graph-searchable embeddings database."""

  @classmethod
  def create(cls, **kwargs):
    """Connect to and, if needed, initialize the database."""
    raise NotImplementedError

  def setup(self):
    """Initialize an empty database."""
    raise NotImplementedError

  def count_embeddings(self) -> int:
    """Return a count of all embeddings in the database."""
    raise NotImplementedError

  def count_edges(self) -> int:
    """Return a count of all edges in the database."""
    raise NotImplementedError

  def insert_edge(self, x_id: int, y_id: int) -> None:
    """Add a directed edge from x_id to y_id."""
    raise NotImplementedError

  def delete_edge(self, x_id, y_id) -> None:
    """Delete an edge between two embeddings."""
    raise NotImplementedError

  def delete_edges(self, x_id) -> None:
    """Delete all edges originating from x_id."""
    raise NotImplementedError

  def drop_all_edges(self) -> None:
    """Delete all edges in the database."""
    raise NotImplementedError

  def insert_embedding(self, embedding, **metadata) -> int:
    """Add an embedding with metadata to the database."""
    raise NotImplementedError

  def get_embedding_ids(self) -> tuple[int, ...]:
    """Get all embedding IDs in the database."""
    raise NotImplementedError

  def get_embedding(self, embedding_id: int) -> np.ndarray:
    """Retrieve an embedding from the database."""
    raise NotImplementedError

  def get_embeddings(self, embedding_ids: Sequence[int]) -> np.ndarray:
    """Get an array of embeddings for the indicated IDs."""
    raise NotImplementedError

  def get_metadata(self, embedding_id: int) -> dict[str, Any]:
    raise NotImplementedError

  def get_edges(self, embedding_id: int) -> tuple[int, ...]:
    raise NotImplementedError

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    raise NotImplementedError
