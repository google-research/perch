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

"""Conversion for TFRecord embeddings to Hoplite DB."""

import os
from chirp.inference import embed_lib
from chirp.inference import tf_examples
from chirp.projects.agile2 import embed
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from etils import epath
import numpy as np
import tqdm


def convert_tfrecords(
    embeddings_path: str,
    db_type: str,
    dataset_name: str,
    max_count: int = -1,
    **kwargs,
):
  """Convert a TFRecord embeddings dataset to a Hoplite DB."""
  ds = tf_examples.create_embeddings_dataset(
      embeddings_path,
      'embeddings-*',
  )
  # Peek at one embedding to get the embedding dimension.
  for ex in ds.as_numpy_iterator():
    emb_dim = ex['embedding'].shape[-1]
    break
  else:
    raise ValueError('No embeddings found.')

  if db_type == 'sqlite':
    db_path = kwargs['db_path']
    if epath.Path(db_path).exists():
      raise ValueError(f'DB path {db_path} already exists.')
    db = sqlite_impl.SQLiteGraphSearchDB.create(db_path, embedding_dim=emb_dim)
  elif db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(
        embedding_dim=emb_dim,
        max_size=kwargs['max_size'],
        degree_bound=kwargs['degree_bound'],
    )
  else:
    raise ValueError(f'Unknown db type: {db_type}')
  db.setup()

  # Convert embedding config to new format and insert into the DB.
  legacy_config = embed_lib.load_embedding_config(embeddings_path)
  model_config = embed.ModelConfig(
      model_key=legacy_config.embed_fn_config.model_key,
      model_config=legacy_config.embed_fn_config.model_config,
  )
  file_id_depth = legacy_config.embed_fn_config['file_id_depth']
  audio_globs = []
  for glob in legacy_config.source_file_patterns:
    new_glob = glob.split('/')[-file_id_depth - 1 :]
    audio_globs.append(new_glob)

  embed_config = embed.EmbedConfig(
      audio_globs={dataset_name: tuple(audio_globs)},
      min_audio_len_s=legacy_config.embed_fn_config.min_audio_s,
      target_sample_rate_hz=legacy_config.embed_fn_config.get(
          'target_sample_rate_hz', -1
      ),
  )
  db.insert_metadata('legacy_config', legacy_config)
  db.insert_metadata('embed_config', embed_config.to_config_dict())
  db.insert_metadata('model_config', model_config.to_config_dict())
  hop_size_s = model_config.model_config.hop_size_s

  for ex in tqdm.tqdm(ds.as_numpy_iterator()):
    embs = ex['embedding']
    print(embs.shape)
    flat_embeddings = np.reshape(embs, [-1, embs.shape[-1]])
    file_id = str(ex['filename'], 'utf8')
    offset_s = ex['timestamp_s']
    if max_count > 0 and db.count_embeddings() >= max_count:
      break
    for i in range(flat_embeddings.shape[0]):
      embedding = flat_embeddings[i]
      offset = np.array(offset_s + hop_size_s * i)
      source = interface.EmbeddingSource(dataset_name, file_id, offset)
      db.insert_embedding(embedding, source)
      if max_count > 0 and db.count_embeddings() >= max_count:
        break
  db.commit()
  num_embeddings = db.count_embeddings()
  print('\n\nTotal embeddings : ', num_embeddings)
  hours_equiv = num_embeddings / 60 / 60 * hop_size_s
  print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')
  return db
