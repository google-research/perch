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

"""SQLite hoplite impliementation using usearch for vector storage."""

import collections
from collections.abc import Sequence
import dataclasses
import json
import sqlite3
from typing import Any

from chirp.projects.hoplite import interface
from etils import epath
from ml_collections import config_dict
import numpy as np
from usearch import index as uindex


USEARCH_CONFIG_KEY = 'usearch_config'
EMBEDDINGS_TABLE = 'hoplite_embeddings'
EDGES_TABLE = 'hoplite_edges'

HOPLITE_FILENAME = 'hoplite.sqlite'
UINDEX_FILENAME = 'usearch.index'

USEARCH_DTYPES = {
    'float16': uindex.ScalarKind.F16,
}


@dataclasses.dataclass
class SQLiteUsearchDB(interface.GraphSearchDBInterface):
  """SQLite hoplite database, using USearch for vector storage.

  USearch provides both indexing for approximate nearest neighbor search and
  fast disk-based random access to vectors for the complete database.

  To handle this, we maintain two USearch indexes:
  1. An in-memory index, which is used when adding new embeddings.
  2. A disk-based index, which is used for random access to the entire database.

  We leave the in-memory index unpopulated until a new embedding is added,
  preferring to use the disk-based index. Once an embedding is added, the
  in-memory index is populated and used for all subsequent operations. On
  commit, the in-memory index is persisted to disk.

  Attributes:
    db_path: The path to the database file.
    db: The sqlite3 database connection.
    ui: The USearch index. Points to either _ui_mem or _ui_disk_view, depending
      on whether new embeddings have been added.
    _ui_mem: The in-memory USearch index used for building the index.
    _ui_disk_view: A disk view of the USearch index.
    embedding_dim: The dimension of the embeddings.
    embedding_dtype: The data type of the embeddings.
    _cursor: The sqlite3 cursor.
  """

  # User-provided.
  db_path: str

  # Instantiated during creation.
  db: sqlite3.Connection
  ui: uindex.Index
  _ui_mem: uindex.Index
  _ui_disk_view: uindex.Index

  # Obtained from the usearch_cfg.
  embedding_dim: int
  embedding_dtype: type[Any] = np.float16

  # Dynamic state.
  _cursor: sqlite3.Cursor | None = None

  @classmethod
  def create(
      cls,
      db_path: str,
      usearch_cfg: config_dict.ConfigDict | None = None,
  ):
    db_path = epath.Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    hoplite_path = db_path / HOPLITE_FILENAME
    db = sqlite3.connect(hoplite_path.as_posix())
    cursor = db.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode
    _setup_sqlite_tables(cursor)
    db.commit()

    # TODO(tomdenton): Check that config is consistent with the DB.
    metadata = _get_metadata(cursor, None)
    if (USEARCH_CONFIG_KEY in metadata
        and usearch_cfg is not None
        and metadata[USEARCH_CONFIG_KEY] != usearch_cfg):
      raise ValueError(
          'A usearch_cfg was provided, but one already exists in the DB.')
    elif USEARCH_CONFIG_KEY in metadata:
      usearch_cfg = metadata[USEARCH_CONFIG_KEY]
    elif usearch_cfg is None:
      raise ValueError('No usearch config found in DB and none provided.')
    else:
      _insert_metadata(cursor, USEARCH_CONFIG_KEY, usearch_cfg)
      db.commit()

    usearch_dtype = USEARCH_DTYPES[usearch_cfg.dtype]
    ui_mem = uindex.Index(
        ndim=usearch_cfg.embedding_dim,
        metric=getattr(uindex.MetricKind, usearch_cfg.metric_name),
        expansion_add=usearch_cfg.expansion_add,
        expansion_search=usearch_cfg.expansion_search,
        dtype=usearch_dtype,
    )
    index_path = db_path / UINDEX_FILENAME
    ui_disk_view = uindex.Index(ndim=usearch_cfg.embedding_dim)
    if index_path.exists():
      ui_disk_view.view(index_path.as_posix())
      ui = ui_disk_view
    else:
      ui = ui_mem

    return SQLiteUsearchDB(
        db_path=db_path.as_posix(),
        db=db,
        ui=ui,
        _ui_mem=ui_mem,
        _ui_disk_view=ui_disk_view,
        embedding_dim=usearch_cfg.embedding_dim,
        embedding_dtype=usearch_cfg.dtype,
    )

  @property
  def _sqlite_filepath(self) -> epath.Path:
    return epath.Path(self.db_path) / HOPLITE_FILENAME

  @property
  def _usearch_filepath(self) -> epath.Path:
    return epath.Path(self.db_path) / UINDEX_FILENAME

  def thread_split(self):
    """Get a new instance of the SQLite DB."""
    return self.create(self.db_path)

  def _get_cursor(self) -> sqlite3.Cursor:
    if self._cursor is None:
      self._cursor = self.db.cursor()
    return self._cursor

  def serialize_edges(self, edges: np.ndarray) -> bytes:
    return edges.astype(np.dtype(np.int64).newbyteorder('<')).tobytes()

  def deserialize_edges(self, serialized_edges: bytes) -> np.ndarray:
    return np.frombuffer(
        serialized_edges,
        dtype=np.dtype(np.int64).newbyteorder('<'),
    )

  def commit(self) -> None:
    self.db.commit()
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None
    if self.ui.size > self._ui_disk_view.size:
      # We have added something to the in-memory index, so persist to disk.
      # This check is sufficient because the index is strictly additive.
      self.ui.save(self._usearch_filepath.as_posix())

  def vacuum_db(self) -> None:
    """Clears out the WAL log and defragments data."""
    cursor = self._get_cursor()
    cursor.execute('VACUUM;')
    self.db.commit()
    cursor.close()
    self._cursor = None
    self.db.close()
    self.db = sqlite3.connect(self.db_path)

  def get_embedding_ids(self) -> np.ndarray:
    # Note that USearch can also create a list of all keys, but it seems
    # quite slow.
    cursor = self._get_cursor()
    cursor.execute("""SELECT id FROM hoplite_embeddings;""")
    return np.array(tuple(int(c[0]) for c in cursor.fetchall()))

  def get_one_embedding_id(self) -> int:
    cursor = self._get_cursor()
    cursor.execute("""SELECT id FROM hoplite_embeddings LIMIT 1;""")
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
    cursor = self._get_cursor()
    return _get_metadata(cursor, key)

  def get_dataset_names(self) -> tuple[str, ...]:
    """Get all dataset names in the database."""
    cursor = self._get_cursor()
    cursor.execute("""SELECT DISTINCT dataset FROM hoplite_sources;""")
    return tuple(c[0] for c in cursor.fetchall())

  def insert_edges(
      self, x_id: int, y_ids: np.ndarray, replace: bool = False
  ) -> None:
    cursor = self._get_cursor()
    if not replace:
      existing = self.get_edges(x_id)
      y_ids = np.unique(np.concatenate([existing, y_ids], axis=0))
    cursor.execute(
        """
        REPLACE INTO hoplite_edges (source_embedding_id, target_embedding_ids)
        VALUES (?, ?);
        """,
        (int(x_id), self.serialize_edges(y_ids)),
    )

  def insert_edge(self, x_id: int, y_id: int):
    self.insert_edges(x_id, np.array([y_id]))

  def delete_edge(self, x_id: int, y_id: int):
    existing = self.get_edges(x_id)
    new_edges = existing[existing != y_id]
    self.insert_edges(x_id, new_edges, replace=True)

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
    source_id = self._get_source_id(
        source.dataset_name, source.source_id, insert=True
    )
    offset_bytes = serialize_embedding(source.offsets, self.embedding_dtype)
    cursor.execute(
        """
      INSERT INTO hoplite_embeddings (source_idx, offsets) VALUES (?, ?);
    """,
        (source_id, offset_bytes),
    )
    embedding_id = int(cursor.lastrowid)

    if self._ui_mem.size == 0 and self._ui_disk_view.size > 0:
      # We need to load the disk view into memory.
      self._ui_mem.load(self._usearch_filepath.as_posix())
      self.ui = self._ui_mem

    self.ui.add(embedding_id, embedding.astype(self.embedding_dtype))
    return embedding_id

  def count_embeddings(self) -> int:
    """Counts the number of hoplite_embeddings in the 'embeddings' table."""
    return self.ui.size

  def embedding_dimension(self) -> int:
    return self.embedding_dim

  def get_embedding(self, embedding_id: int) -> np.ndarray:
    contains = self.ui.contains(embedding_id)
    if not np.all(contains):
      raise ValueError(f'Embeddings {embedding_id} not found.')
    emb = self.ui.get(embedding_id)
    return np.array(emb)

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
    offsets = deserialize_embedding(offsets, self.embedding_dtype)
    return interface.EmbeddingSource(dataset, str(source), offsets)

  def get_embeddings(
      self, embedding_ids: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    contains = self.ui.contains(embedding_ids)
    if not np.all(contains):
      raise ValueError(f'Embeddings {embedding_ids[~contains]} not found.')
    embs = self.ui.get(embedding_ids)
    return embedding_ids, np.array(embs)

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
      source_idx = self._get_source_id(dataset_name, source_id, insert=False)
      query = (
          'SELECT id, offsets FROM hoplite_embeddings '
          'WHERE hoplite_embeddings.source_idx = ?;'
      )
      cursor.execute(query, (source_idx,))
    result_pairs = cursor.fetchall()
    outputs = []
    for idx, offsets_bytes in result_pairs:
      got_offsets = deserialize_embedding(offsets_bytes, self.embedding_dtype)
      if offsets is not None and not np.array_equal(got_offsets, offsets):
        continue
      outputs.append(idx)
    return np.array(outputs)

  def get_edges(self, embedding_id: int) -> np.ndarray:
    query = (
        'SELECT hoplite_edges.target_embedding_ids FROM hoplite_edges '
        'WHERE hoplite_edges.source_embedding_id = ?;'
    )
    cursor = self._get_cursor()
    cursor.execute(
        query,
        (int(embedding_id),),
    )
    edge_bytes = cursor.fetchall()
    if not edge_bytes:
      return np.array([])
    return self.deserialize_edges(edge_bytes[0][0])

  def insert_label(
      self, label: interface.Label, skip_duplicates: bool = False
  ) -> bool:
    if label.type is None:
      raise ValueError('label type must be set')
    if label.provenance is None:
      raise ValueError('label source must be set')
    if skip_duplicates and label in self.get_labels(label.embedding_id):
      return False

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
    return True

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

  def get_classes(self) -> Sequence[str]:
    cursor = self._get_cursor()
    cursor.execute('SELECT DISTINCT label FROM hoplite_labels ORDER BY label;')
    return tuple(r[0] for r in cursor.fetchall())

  def get_class_counts(
      self, label_type: interface.LabelType = interface.LabelType.POSITIVE
  ) -> dict[str, int]:
    cursor = self._get_cursor()
    # Subselect with distinct is needed to avoid double-counting the same label
    # on the same embedding because of different provenances.
    cursor.execute("""
      SELECT label, type, COUNT(*)
      FROM (
          SELECT DISTINCT embedding_id, label, type FROM hoplite_labels)
      GROUP BY label, type;
    """)
    results = collections.defaultdict(int)
    for r in cursor.fetchall():
      if r[1] == label_type.value:
        results[r[0]] = r[2]
      else:
        results[r[0]] += 0
    return results

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


def _setup_sqlite_tables(cursor: sqlite3.Cursor) -> None:
  """Create the needed tables in the SQLite databse.

  This is similar to the basice SQLite implementation, but we do not need an
  edges table because the USearch index is self-contained. We also do not need
  to store the embedding in the embeddings table, though we do still need a
  mapping of embedding ID to source ID and offsets.

  Args:
    cursor: The SQLite cursor to use.
  """
  cursor.execute("""
  SELECT name FROM sqlite_master WHERE type='table' AND name='hoplite_labels';
  """)
  if cursor.fetchone() is not None:
    return

  # Create embedding sources table
  cursor.execute("""
      CREATE TABLE IF NOT EXISTS hoplite_sources (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          dataset STRING NOT NULL,
          source STRING NOT NULL
      );
  """)

  # Create embeddings table for joining embeddings with sources.
  cursor.execute("""
      CREATE TABLE IF NOT EXISTS hoplite_embeddings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
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
      target_embedding_ids BLOB NOT NULL,
      FOREIGN KEY (source_embedding_id) REFERENCES embeddings(id)
  );
  """)

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
  CREATE UNIQUE INDEX IF NOT EXISTS idx_source_embedding ON hoplite_edges (source_embedding_id);
  """)
  cursor.execute("""
  CREATE INDEX IF NOT EXISTS idx_label ON hoplite_labels (embedding_id, label);
  """)


def _insert_metadata(
    cursor: sqlite3.Cursor, key: str, value: config_dict.ConfigDict
):
  """Insert a key-value pair into the metadata table."""
  json_coded = value.to_json()
  cursor.execute(
      """
    INSERT INTO hoplite_metadata (key, data) VALUES (?, ?)
    ON CONFLICT (key) DO UPDATE SET data = excluded.data;
  """,
      (key, json_coded),
  )


def _get_metadata(cursor: sqlite3.Cursor, key: str | None):
  """Retrieve metadata from the SQLite database."""
  if key is None:
    cursor.execute("""SELECT key, data FROM hoplite_metadata;""")
    return config_dict.ConfigDict(
        {k: json.loads(v) for k, v in cursor.fetchall()}
    )

  cursor.execute(
      """
    SELECT data FROM hoplite_metadata WHERE key = ?;
  """,
      (key,),
  )
  result = cursor.fetchone()
  if result is None:
    raise KeyError(f'Metadata key not found: {key}')
  return config_dict.ConfigDict(json.loads(result[0]))


def serialize_embedding(
    embedding: np.ndarray, embedding_dtype: type[Any]
) -> bytes:
  return embedding.astype(np.dtype(embedding_dtype).newbyteorder('<')).tobytes()


def deserialize_embedding(
    serialized_embedding: bytes, embedding_dtype: type[Any]
) -> np.ndarray:
  return np.frombuffer(
      serialized_embedding,
      dtype=np.dtype(embedding_dtype).newbyteorder('<'),
  )
