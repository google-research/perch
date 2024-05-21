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

from collections.abc import Sequence
import dataclasses
import sqlite3
from chirp.projects.hoplite import interface
import numpy as np


@dataclasses.dataclass
class SQLiteGraphSearchDB(interface.GraphSearchDBInterface):
  """SQLite implementation of graph search database."""

  db: sqlite3.Connection
  embedding_dtype: str = 'float16'
  _cursor: sqlite3.Cursor | None = None

  @classmethod
  def create(cls, db_path: str):
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode
    db.commit()
    return SQLiteGraphSearchDB(db)

  def _get_cursor(self) -> sqlite3.Cursor:
    if self._cursor is None:
      self._cursor = self.db.cursor()
    return self._cursor

  def serialize_embedding(self, embedding: np.ndarray) -> bytes:
    # TODO(tomdenton): Consider using protobufs.
    return embedding.astype(self.embedding_dtype).tobytes()

  def deserialize_embedding(self, serialized_embedding: bytes) -> np.ndarray:
    # TODO(tomdenton): Consider using protobufs.
    return np.frombuffer(serialized_embedding, dtype=self.embedding_dtype)

  def setup(self, index=True):
    cursor = self._get_cursor()
    # Create embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL
        );
    """)

    # Create metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding_id INTEGER,
            file_id TEXT,
            timestamp_s REAL NOT NULL,
            FOREIGN KEY(embedding_id) REFERENCES embeddings(id)
        );
    """)

    # Create edges table.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_embedding_id INTEGER NOT NULL,
        target_embedding_id INTEGER NOT NULL,
        FOREIGN KEY (source_embedding_id) REFERENCES embeddings(id),
        FOREIGN KEY (target_embedding_id) REFERENCES embeddings(id)
    )""")

    if index:
      # Create indices for efficient lookups.
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_source_embedding ON edges (source_embedding_id);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_target_embedding ON edges (target_embedding_id);
      """)
      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_embedding_id ON metadata (embedding_id);
      """)
    self.db.commit()

  def commit(self):
    self.db.commit()

  def get_embedding_ids(self) -> tuple[int, ...]:
    cursor = self._get_cursor()
    cursor.execute("""SELECT id FROM embeddings;""")
    return tuple(c[0] for c in cursor.fetchall())

  def insert_edge(self, x_id: int, y_id: int):
    if not isinstance(x_id, int):
      raise ValueError('x_id must be an integer')
    if not isinstance(y_id, int):
      raise ValueError('y_id must be an integer')
    cursor = self._get_cursor()
    cursor.execute(
        """
      INSERT INTO edges (source_embedding_id, target_embedding_id) VALUES (?, ?);
    """,
        (x_id, y_id),
    )

  def delete_edge(self, x_id: int, y_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      DELETE FROM edges WHERE source_embedding_id = ? AND target_embedding_id = ?;
    """,
        (x_id, y_id),
    )

  def delete_edges(self, x_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      DELETE FROM edges WHERE source_embedding_id = ?;
    """,
        (x_id,),
    )

  def drop_all_edges(self):
    cursor = self._get_cursor()
    cursor.execute("""
      DELETE FROM edges;
    """)
    self.db.commit()

  def insert_embedding(self, embedding: np.ndarray, **metadata) -> int:
    cursor = self._get_cursor()
    embedding_bytes = self.serialize_embedding(embedding)
    cursor.execute(
        """
      INSERT INTO embeddings (embedding) VALUES (?);
    """,
        (embedding_bytes,),
    )
    embedding_id = cursor.lastrowid

    file_id = metadata.get('file_id', '')
    offset_s = metadata.get('offset_s', 0)
    cursor.execute(
        """
      INSERT INTO metadata (embedding_id, file_id, timestamp_s) VALUES (?, ?, ?);
    """,
        (embedding_id, file_id, offset_s),
    )
    return int(embedding_id)

  def count_embeddings(self) -> int:
    """Counts the number of embeddings in the 'embeddings' table."""
    cursor = self._get_cursor()
    cursor.execute('SELECT COUNT(*) FROM embeddings;')
    result = cursor.fetchone()
    return result[0]  # Extract the count from the result tuple

  def count_edges(self) -> int:
    """Counts the number of embeddings in the 'embeddings' table."""
    cursor = self._get_cursor()
    cursor.execute('SELECT COUNT(*) FROM edges;')
    result = cursor.fetchone()
    return result[0]  # Extract the count from the result tuple

  def get_embedding(self, embedding_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT embedding FROM embeddings WHERE id = ?;
    """,
        (embedding_id,),
    )
    embedding = cursor.fetchall()[0][0]
    return self.deserialize_embedding(embedding)

  def get_embeddings(self, embedding_ids: Sequence[int]) -> np.ndarray:
    placeholders = ', '.join('?' * len(embedding_ids))
    query = f'SELECT embedding FROM embeddings WHERE id IN ({placeholders});'
    cursor = self._get_cursor()
    cursor.execute(query, embedding_ids)
    embeddings = cursor.fetchall()
    return np.array(tuple(self.deserialize_embedding(e[0]) for e in embeddings))

  def get_metadata(self, embedding_id: int):
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT metadata.file_id, metadata.timestamp_s FROM metadata WHERE metadata.embedding_id = ?;
    """,
        (embedding_id,),
    )
    return cursor.fetchall()

  def get_edges(self, embedding_id: int) -> tuple[int, ...]:
    cursor = self._get_cursor()
    cursor.execute(
        """
      SELECT edges.target_embedding_id FROM edges WHERE edges.source_embedding_id = ?;
    """,
        (embedding_id,),
    )
    edges = cursor.fetchall()
    return tuple(e[0] for e in edges)

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
