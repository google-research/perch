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

"""SQLite Implementation of a searchable embeddings database."""

import dataclasses
import sqlite3
from typing import Any

from chirp.projects.hoplite import interface
import numpy as np

EMBEDDINGS_TABLE = 'hoplite_embeddings'
EDGES_TABLE = 'hoplite_edges'


@dataclasses.dataclass
class SQLiteGraphSearchDB(interface.GraphSearchDBInterface):
  """SQLite implementation of graph search database."""

  db: sqlite3.Connection
  embedding_dim: int
  embedding_dtype: type = np.float16
  _cursor: sqlite3.Cursor | None = None

  @classmethod
  def create(cls, db_path: str, embedding_dim: int):
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode
    db.commit()
    return SQLiteGraphSearchDB(db, embedding_dim)

  def _get_cursor(self) -> sqlite3.Cursor:
    if self._cursor is None:
      self._cursor = self.db.cursor()
    return self._cursor

  def serialize_embedding(self, embedding: np.ndarray) -> bytes:
    return embedding.astype(
        np.dtype(self.embedding_dtype).newbyteorder('<')
    ).tobytes()

  def deserialize_embedding(self, serialized_embedding: bytes) -> np.ndarray:
    return np.frombuffer(
        serialized_embedding,
        dtype=np.dtype(self.embedding_dtype).newbyteorder('<'),
    )

  def setup(self, index=True):
    cursor = self._get_cursor()
    # Create embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            source STRING NOT NULL
        );
    """)

    # Create hoplite_edges table.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hoplite_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_embedding_id INTEGER NOT NULL,
        target_embedding_id INTEGER NOT NULL,
        FOREIGN KEY (source_embedding_id) REFERENCES embeddings(id),
        FOREIGN KEY (target_embedding_id) REFERENCES embeddings(id)
    )""")

    # Create hoplite_labels table.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hoplite_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding_id INTEGER NOT NULL,
        label STRING NOT NULL,
        type INT NOT NULL,
        source STRING NOT NULL,
        FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
    )""")

    if index:
      # Create indices for efficient lookups.
      cursor.execute("""
      CREATE UNIQUE INDEX IF NOT EXISTS
          idx_embedding ON hoplite_embeddings (id);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS embedding_source ON hoplite_embeddings (source);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_source_embedding ON hoplite_edges (source_embedding_id);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_target_embedding ON hoplite_edges (target_embedding_id);
      """)
      cursor.execute("""
      CREATE UNIQUE INDEX IF NOT EXISTS
          idx_edge ON hoplite_edges (source_embedding_id, target_embedding_id);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_label ON hoplite_labels (embedding_id, label);
      """)
    self.db.commit()

  def commit(self):
    self.db.commit()

  def get_embedding_ids(self) -> np.ndarray:
    cursor = self._get_cursor()
    cursor.execute("""SELECT id FROM hoplite_embeddings;""")
    return np.array(tuple(int(c[0]) for c in cursor.fetchall()))

  def get_one_embedding_id(self) -> int:
    cursor = self._get_cursor()
    cursor.execute("""SELECT id FROM hoplite_embeddings;""")
    return int(cursor.fetchone())

  def insert_edge(self, x_id: int, y_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      INSERT INTO hoplite_edges (source_embedding_id, target_embedding_id) VALUES (?, ?);
    """,
        (int(x_id), int(y_id)),
    )

  def delete_edge(self, x_id: int, y_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      DELETE FROM hoplite_edges WHERE source_embedding_id = ? AND target_embedding_id = ?;
    """,
        (int(x_id), int(y_id)),
    )

  def delete_edges(self, x_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      DELETE FROM hoplite_edges WHERE source_embedding_id = ?;
    """,
        (int(x_id),),
    )

  def drop_all_edges(self):
    cursor = self._get_cursor()
    cursor.execute("""
      DELETE FROM hoplite_edges;
    """)
    self.db.commit()

  def insert_embedding(self, embedding: np.ndarray, source: str) -> int:
    if embedding.shape[-1] != self.embedding_dim:
      raise ValueError('Incorrect embedding dimension.')
    cursor = self._get_cursor()
    embedding_bytes = self.serialize_embedding(embedding)
    cursor.execute(
        """
      INSERT INTO hoplite_embeddings (embedding, source) VALUES (?, ?);
    """,
        (embedding_bytes, source),
    )
    embedding_id = cursor.lastrowid
    return int(embedding_id)

  def count_embeddings(self) -> int:
    """Counts the number of hoplite_embeddings in the 'embeddings' table."""
    cursor = self._get_cursor()
    cursor.execute('SELECT COUNT(*) FROM hoplite_embeddings;')
    result = cursor.fetchone()
    return result[0]  # Extract the count from the result tuple

  def embedding_dimension(self) -> int:
    return self.embedding_dim

  def count_edges(self) -> int:
    """Counts the number of hoplite_embeddings in the 'embeddings' table."""
    cursor = self._get_cursor()
    cursor.execute('SELECT COUNT(*) FROM hoplite_edges;')
    result = cursor.fetchone()
    return result[0]  # Extract the count from the result tuple

  def get_embedding(self, embedding_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT embedding FROM hoplite_embeddings WHERE id = ?;
    """,
        (int(embedding_id),),
    )
    embedding = cursor.fetchall()[0][0]
    return self.deserialize_embedding(embedding)

  def get_embedding_source(self, embedding_id: int) -> str:
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT source FROM hoplite_embeddings WHERE id = ?;
    """,
        (int(embedding_id),),
    )
    return cursor.fetchall()[0][0]

  def get_embeddings(
      self, embedding_ids: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    placeholders = ', '.join('?' * embedding_ids.shape[0])
    query = (
        'SELECT id, embedding FROM hoplite_embeddings WHERE id IN '
        f'({placeholders}) ORDER BY id;'
    )
    cursor = self._get_cursor()
    results = cursor.execute(
        query, tuple(int(c) for c in embedding_ids)
    ).fetchall()
    result_ids = np.array(tuple(int(c[0]) for c in results))
    embeddings = np.array(
        tuple(self.deserialize_embedding(c[1]) for c in results)
    )
    return result_ids, embeddings

  def get_embeddings_by_source(self, source: str) -> np.ndarray:
    """Get the embedding IDs for all embeddings matching the source."""
    query = (
        'SELECT id FROM hoplite_embeddings WHERE hoplite_embeddings.source = ?;'
    )
    cursor = self._get_cursor()
    cursor.execute(query, (source,))
    idxes = cursor.fetchall()
    return np.array(tuple(e[0] for e in idxes))

  def get_edges(self, embedding_id: int) -> np.ndarray:
    query = (
        'SELECT hoplite_edges.target_embedding_id FROM hoplite_edges '
        'WHERE hoplite_edges.source_embedding_id = ?;'
    )
    cursor = self._get_cursor()
    cursor.execute(
        query,
        (int(embedding_id),),
    )
    edges = cursor.fetchall()
    return np.array(tuple(e[0] for e in edges))

  def insert_label(self, label: interface.Label):
    if label.type is None:
      raise ValueError('label type must be set')
    if label.label_source is None:
      raise ValueError('label source must be set')
    cursor = self._get_cursor()
    cursor.execute(
        """
      INSERT INTO hoplite_labels (embedding_id, label, type, source) VALUES (?, ?, ?, ?);
    """,
        (
            int(label.embedding_id),
            label.label,
            label.type.value,
            label.label_source,
        ),
    )

  def get_embeddings_by_label(
      self,
      label: str,
      label_type: interface.LabelType | None = interface.LabelType.POSITIVE,
      label_source: str | None = None,
  ) -> np.ndarray:
    query = 'SELECT embedding_id FROM hoplite_labels WHERE label = ?'
    pred = (label,)
    if label_type is not None:
      query += ' AND type = ?'
      pred = pred + (label_type.value,)
    if label_source is not None:
      query += ' AND source = ?'
      pred = pred + (label_source,)
    cursor = self._get_cursor()
    results = cursor.execute(query, pred).fetchall()
    ids = np.array(tuple(int(c[0]) for c in results), np.int64)
    return np.unique(ids)

  def get_labels(self, embedding_id: int) -> tuple[interface.Label, ...]:
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT embedding_id, label, type, source FROM hoplite_labels WHERE embedding_id = ?;
    """,
        (int(embedding_id),),
    )
    results = cursor.fetchall()
    return tuple(
        interface.Label(int(r[0]), r[1], interface.LabelType(r[2]), r[3])
        for r in results
    )

  def print_table_values(self, table_name):
    """Prints all values from the specified table in the SQLite database."""
    cursor = self._get_cursor()
    select_query = f'SELECT * FROM {table_name};'  # Query to select all rows
    cursor.execute(select_query)

    # Fetch all rows
    rows = cursor.fetchall()

    # Print header (optional)
    if rows:
      # Column names
      print(', '.join(column[0] for column in cursor.description))

    # Print each row as a comma-separated string
    for row in rows:
      print(', '.join(str(value) for value in row))
