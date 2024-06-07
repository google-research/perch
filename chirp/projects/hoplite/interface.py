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

"""Base class for a searchable embeddings database."""

import abc
import dataclasses
import enum
from typing import Sequence

import numpy as np


class LabelType(enum.Enum):
  NEGATIVE = 0
  POSITIVE = 1


@dataclasses.dataclass
class Label:
  """Label for an embedding.

  Attributes:
    embedding_id: Unique integer ID for the embedding this label applies to.
    label: Label string.
    type: Type of label (positive, negative, etc).
    label_source: Freeform field describing where the label came from (eg,
      labeler name, model identifier for pseudolabels, etc).
  """

  embedding_id: int
  label: str
  type: LabelType
  label_source: str


class GraphSearchDBInterface(abc.ABC):
  """Interface for graph-searchable embeddings database.

  The database consists of a table of embeddings with a metadata string and
  a unique id for each embedding. The IDs are chosen by the database when the
  embedding is inserted. The database also contains a tabel of `Label`s,
  and (as needed) a table of graph edges facilitating faster search.

  Methods are split into 'Base' methods and 'Composite' methods. Base methods
  must be implemented for any implementation. Composite methods have a default
  implementation using the base methods, but may benefit from implementation-
  specific optimizations.
  """

  # Base methods

  @classmethod
  @abc.abstractmethod
  def create(cls, **kwargs):
    """Connect to and, if needed, initialize the database."""

  @abc.abstractmethod
  def setup(self):
    """Initialize an empty database."""

  @abc.abstractmethod
  def commit(self) -> None:
    """Commit any pending transactions to the database."""

  @abc.abstractmethod
  def get_embedding_ids(self) -> np.ndarray:
    # TODO(tomdenton): Make this return an iterator, with optional shuffling.
    """Get all embedding IDs in the database."""

  @abc.abstractmethod
  def insert_embedding(self, embedding: np.ndarray, source: str) -> int:
    """Add an embedding to the database."""

  @abc.abstractmethod
  def get_embedding(self, embedding_id: int) -> np.ndarray:
    """Retrieve an embedding from the database."""

  @abc.abstractmethod
  def get_embedding_source(self, embedding_id: int) -> str:
    """Get the source corresponding to the given embedding_id."""

  @abc.abstractmethod
  def get_embeddings_by_source(self, source: str) -> np.ndarray:
    """Get the embedding IDs for all embeddings matching the source."""

  @abc.abstractmethod
  def insert_edge(self, x_id: int, y_id: int) -> None:
    """Add a directed edge from x_id to y_id."""

  @abc.abstractmethod
  def delete_edge(self, x_id, y_id) -> None:
    """Delete an edge between two embeddings."""

  @abc.abstractmethod
  def get_edges(self, embedding_id: int) -> np.ndarray:
    """Get all embedding_id's adjacent to the target embedding_id."""

  @abc.abstractmethod
  def insert_label(self, label: Label) -> None:
    """Add a label to the db."""

  @abc.abstractmethod
  def embedding_dimension(self) -> int:
    """Get the embedding dimension."""

  @abc.abstractmethod
  def get_embeddings_by_label(
      self,
      label: str,
      label_type: LabelType | None = LabelType.POSITIVE,
      label_source: str | None = None,
  ) -> np.ndarray:
    """Find embeddings by label.

    Args:
      label: Label string to search for.
      label_type: Type of label to return. If None, returns all labels
        regardless of Type.
      label_source: If provided, filters to the target label_source value.

    Returns:
      An array of unique embedding id's matching the label.
    """

  @abc.abstractmethod
  def get_labels(self, embedding_id: int) -> Sequence[Label]:
    """Get all labels for the indicated embedding_id."""

  # Composite methods

  def get_one_embedding_id(self) -> int:
    """Get an arbitrary embedding id from the database."""
    return self.get_embedding_ids()[0]

  def count_embeddings(self) -> int:
    """Return a count of all embeddings in the database."""
    return len(self.get_embedding_ids())

  def count_edges(self) -> int:
    """Return a count of all edges in the database."""
    ct = 0
    for idx in self.get_embedding_ids():
      ct += self.get_edges(idx).shape[0]
    return ct

  def get_embeddings(
      self, embedding_ids: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Get an array of embeddings for the indicated IDs.

    Note that the embeddings may not be returned in the same order as the
    provided embedding_id's. Thus, we suggest the usage:
    ```
    idxes, embeddings = db.get_embeddings(idxes)
    ```

    Args:
      embedding_ids: 1D array of embedding id's.

    Returns:
      Permuted array of embedding_id's and embeddings.
    """
    embeddings = [self.get_embedding(int(idx)) for idx in embedding_ids]
    return embedding_ids, np.array(embeddings)

  def insert_edges(self, x_id: int, y_ids: np.ndarray) -> None:
    """Add a set of directed edges from x_id to each id in y_ids."""
    for y_id in y_ids:
      self.insert_edge(x_id, int(y_id))

  def drop_all_edges(self) -> None:
    """Delete all edges in the database."""
    for idx in self.get_embedding_ids():
      self.delete_edges(idx)

  def delete_edges(self, x_id) -> None:
    """Delete all edges originating from x_id."""
    nbrs = self.get_edges(x_id)
    for nbr in nbrs:
      self.delete_edge(x_id, nbr)
