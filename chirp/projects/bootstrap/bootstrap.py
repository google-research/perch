# coding=utf-8
# Copyright 2023 The Chirp Authors.
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

"""Configuration and init library for Search Bootstrap projects."""

import dataclasses
from typing import Sequence

from chirp.inference import embed_lib
from chirp.inference import interface
from chirp.inference import models
from chirp.inference import tf_examples
from etils import epath
from ml_collections import config_dict
import tensorflow as tf


@dataclasses.dataclass
class BootstrapState:
  """Union of data and models useful to go from a few examples to a detector."""

  config: 'BootstrapConfig'
  embedding_model: interface.EmbeddingModel | None = None
  embeddings_dataset: tf.data.Dataset | None = None
  source_map: dict[str, embed_lib.SourceInfo] | None = None

  def __post_init__(self):
    if self.embedding_model is None:
      self.embedding_model = models.model_class_map()[
          self.config.model_key
      ].from_config(self.config.model_config)
    self.create_embeddings_dataset()
    self.create_source_map()

  def create_embeddings_dataset(self):
    """Create a TF Dataset of the embeddings."""
    if self.embeddings_dataset:
      return self.embeddings_dataset
    ds = tf_examples.create_embeddings_dataset(
        self.config.embeddings_path, 'embeddings-*'
    )
    self.embeddings_dataset = ds
    return ds

  def create_source_map(self):
    """Map filenames to full filepaths."""
    if self.config.audio_globs is None:
      raise ValueError('Cannot create source map with no audio globs.')
    source_infos = embed_lib.create_source_infos(self.config.audio_globs, 1, -1)

    self.source_map = {}
    for s in source_infos:
      file_id = epath.Path(
          *epath.Path(s.filepath).parts[-(self.config.file_id_depth + 1) :]
      ).as_posix()
      dupe = self.source_map.get(file_id)
      if dupe:
        raise ValueError(
            'All base filenames must be unique. '
            f'Filename {file_id} appears in both {s.filepath} and {dupe}.'
        )
      self.source_map[file_id] = s.filepath


@dataclasses.dataclass
class BootstrapConfig:
  """Configuration for Search Bootstrap project."""

  # Embeddings dataset info.
  embeddings_path: str

  # Annotations info.
  annotated_path: str

  # The following are populated automatically from the embedding config.
  embedding_hop_size_s: float | None = None
  file_id_depth: int | None = None
  audio_globs: Sequence[str] | None = None
  model_key: str | None = None
  model_config: config_dict.ConfigDict | None = None

  @classmethod
  def load_from_embedding_config(
      cls, embeddings_path: str, annotated_path: str
  ):
    """Instantiate from a configuration written alongside embeddings."""
    embedding_config = embed_lib.load_embedding_config(embeddings_path)
    embed_fn_config = embedding_config.embed_fn_config

    # Extract the embedding model config from the embedding_config.
    if embed_fn_config.model_key == 'separate_embed_model':
      # If a separation model was applied, get the embedding model config only.
      model_key = 'taxonomy_model_tf'
      model_config = embed_fn_config.model_config.taxonomy_model_tf_config
    else:
      model_key = embed_fn_config.model_key
      model_config = embed_fn_config.model_config
    return BootstrapConfig(
        embeddings_path=embeddings_path,
        annotated_path=annotated_path,
        model_key=model_key,
        model_config=model_config,
        embedding_hop_size_s=model_config.hop_size_s,
        file_id_depth=embed_fn_config.file_id_depth,
        audio_globs=embedding_config.source_file_patterns,
    )
