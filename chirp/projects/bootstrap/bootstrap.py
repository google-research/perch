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
      self.embedding_model = models.model_class_map()[self.config.model_key](
          **self.config.model_config
      )
    self.create_embeddings_dataset()
    self.create_source_map()

  def create_embeddings_dataset(self):
    """Create a TF Dataset of the embeddings."""
    if self.embeddings_dataset:
      return self.embeddings_dataset
    if '*' not in self.config.embeddings_glob:
      ds = tf_examples.create_embeddings_dataset(self.config.embeddings_glob)
    else:
      # find the first segment with a *.
      dirs = self.config.embeddings_glob.split('/')
      has_wildcard = ['*' in d for d in dirs]
      first_wildcard = has_wildcard.index(True)
      dirs = '/'.join(dirs[:first_wildcard])
      glob = '/'.join(dirs[first_wildcard:])
      ds = tf_examples.create_embeddings_dataset(dirs, glob)
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
  embeddings_glob: str
  embedding_hop_size_s: float
  file_id_depth: int
  audio_globs: Sequence[str] | None

  # Annotations info.
  # TODO(tomdenton): Write handling for the annotated data.
  annotated_path: str

  # Model info. Should match the model used for creating embeddings.
  model_key: str
  model_config: config_dict.ConfigDict
