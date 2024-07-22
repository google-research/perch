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
import json
import sqlite3
from typing import Any

from chirp.projects.hoplite import interface
from ml_collections import config_dict
import numpy as np

EMBEDDINGS_TABLE = 'hoplite_embeddings'
EDGES_TABLE = 'hoplite_edges'


@dataclasses.dataclass
class SQLiteGraphSearchDB(interface.GraphSearchDBInterface):
  """SQLite implementation of graph search database."""

  db: sqlite3.Connection
  embedding_dim: int
  embedding_dtype: type[Any] = np.float16
  _cursor: sqlite3.Cursor | None = None

  @classmethod
  def create(
      cls,
      db_path: str,
      embedding_dim: int,
      embedding_dtype: type[Any] = np.float16,
  ):
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
    # Create embedding sources table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset STRING NOT NULL,
            source STRING NOT NULL
        );
    """)

    # Create embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            source_idx INTEGER NOT NULL,
            offsets BLOB NOT NULL,
            FOREIGN KEY (source_idx) REFERENCES hoplite_sources(id)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_metadata (
            key STRING PRIMARY KEY,
            data STRING NOT NULL
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
        provenance STRING NOT NULL,
        FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
    )""")

    if index:
      # Create indices for efficient lookups.
      cursor.execute("""
      CREATE UNIQUE INDEX IF NOT EXISTS
          idx_embedding ON hoplite_embeddings (id);
      """)
      cursor.execute("""
      CREATE UNIQUE INDEX IF NOT EXISTS
          source_pairs ON hoplite_sources (dataset, source);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS embedding_source ON hoplite_embeddings (source_idx);
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
    return int(cursor.fetchone()[0])

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""
    json_coded = value.to_json()
    cursor = self._get_cursor()
    cursor.execute(
        """
      INSERT INTO hoplite_metadata (key, data) VALUES (?, ?)
      ON CONFLICT (key) DO UPDATE SET data = excluded.data;
    """,
        (key, json_coded),
    )

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table."""
    if key is None:
      cursor = self._get_cursor()
      cursor.execute("""SELECT key, data FROM hoplite_metadata;""")
      return config_dict.ConfigDict(
          {k: json.loads(v) for k, v in cursor.fetchall()}
      )

    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT data FROM hoplite_metadata WHERE key = ?;
    """,
        (key,),
    )
    result = cursor.fetchone()
    if result is None:
      raise ValueError(f'Metadata key not found: {key}')
    return config_dict.ConfigDict(json.loads(result[0]))

  def get_dataset_names(self) -> tuple[str, ...]:
    """Get all dataset names in the database."""
    cursor = self._get_cursor()
    cursor.execute("""SELECT DISTINCT dataset FROM hoplite_sources;""")
    return tuple(c[0] for c in cursor.fetchall())

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

  def _get_source_id(
      self, dataset_name: str, source_id: str, insert: bool = False
  ) -> int | None:
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT id FROM hoplite_sources WHERE dataset = ? AND source = ?;
    """,
        (dataset_name, source_id),
    )
    result = cursor.fetchone()
    if result is None and insert:
      cursor.execute(
          """
        INSERT INTO hoplite_sources (dataset, source) VALUES (?, ?);
      """,
          (dataset_name, source_id),
      )
      return int(cursor.lastrowid)
    elif result is None:
      return None
    return int(result[0])

  def insert_embedding(
      self, embedding: np.ndarray, source: interface.EmbeddingSource
  ) -> int:
    if embedding.shape[-1] != self.embedding_dim:
      raise ValueError('Incorrect embedding dimension.')
    cursor = self._get_cursor()
    embedding_bytes = self.serialize_embedding(embedding)
    source_id = self._get_source_id(
        source.dataset_name, source.source_id, insert=True
    )
    offset_bytes = self.serialize_embedding(source.offsets)
    cursor.execute(
        """
      INSERT INTO hoplite_embeddings (embedding, source_idx, offsets) VALUES (?, ?, ?);
    """,
        (embedding_bytes, source_id, offset_bytes),
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

  def get_embedding_source(
      self, embedding_id: int
  ) -> interface.EmbeddingSource:
    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT hs.dataset, hs.source, he.offsets
        FROM hoplite_sources hs
        JOIN hoplite_embeddings he ON hs.id = he.source_idx
        WHERE he.id = ?;
        """,
        (int(embedding_id),),
    )
    dataset, source, offsets = cursor.fetchall()[0]
    offsets = self.deserialize_embedding(offsets)
    return interface.EmbeddingSource(dataset, str(source), offsets)

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

  def get_embeddings_by_source(
      self,
      dataset_name: str,
      source_id: str | None,
      offsets: np.ndarray | None = None,
  ) -> np.ndarray:
    """Get the embedding IDs for all embeddings matching the indicated source.

    Args:
      dataset_name: The name of the dataset to search.
      source_id: The ID of the source to search. If None, all sources are
        searched.
      offsets: The offsets of the source to search. If None, all offsets are
        searched.

    Returns:
      A list of embedding IDs matching the indicated source parameters.
    """
    cursor = self._get_cursor()
    source_id = self._get_source_id(dataset_name, source_id, insert=False)
    if source_id is None:
      query = (
          'SELECT id, offsets FROM hoplite_embeddings '
          'WHERE hoplite_embeddings.source_idx IN ('
          '  SELECT id FROM hoplite_sources '
          '  WHERE hoplite_sources.dataset = ?'
          ');'
      )
      cursor.execute(query, (dataset_name,))
    else:
      query = (
          'SELECT id, offsets FROM hoplite_embeddings '
          'WHERE hoplite_embeddings.source_idx = ?;'
      )
      cursor.execute(query, (source_id,))
    result_pairs = cursor.fetchall()
    outputs = []
    for idx, offsets_bytes in result_pairs:
      got_offsets = self.deserialize_embedding(offsets_bytes)
      if offsets is not None and not np.array_equal(got_offsets, offsets):
        continue
      outputs.append(idx)
    return np.array(outputs)

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
    if label.provenance is None:
      raise ValueError('label source must be set')
    cursor = self._get_cursor()
    cursor.execute(
        """
      INSERT INTO hoplite_labels (embedding_id, label, type, provenance) VALUES (?, ?, ?, ?);
    """,
        (
            int(label.embedding_id),
            label.label,
            label.type.value,
            label.provenance,
        ),
    )

  def get_embeddings_by_label(
      self,
      label: str,
      label_type: interface.LabelType | None = interface.LabelType.POSITIVE,
      provenance: str | None = None,
  ) -> np.ndarray:
    query = 'SELECT embedding_id FROM hoplite_labels WHERE label = ?'
    pred = (label,)
    if label_type is not None:
      query += ' AND type = ?'
      pred = pred + (label_type.value,)
    if provenance is not None:
      query += ' AND provenance = ?'
      pred = pred + (provenance,)
    cursor = self._get_cursor()
    results = cursor.execute(query, pred).fetchall()
    ids = np.array(tuple(int(c[0]) for c in results), np.int64)
    return np.unique(ids)

  def get_labels(self, embedding_id: int) -> tuple[interface.Label, ...]:
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT embedding_id, label, type, provenance FROM hoplite_labels WHERE embedding_id = ?;
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
