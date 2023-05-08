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
import os
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
    embeddings_glob = epath.Path(self.config.embeddings_glob)
    embeddings_files = [fn.as_posix() for fn in embeddings_glob.glob('')]
    ds = tf.data.TFRecordDataset(
        embeddings_files, num_parallel_reads=tf.data.AUTOTUNE
    )

    parser = tf_examples.get_example_parser()
    ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(16)
    self.embeddings_dataset = ds
    return ds

  def create_source_map(self):
    """Map filenames to full filepaths."""
    source_infos = embed_lib.create_source_infos(self.config.audio_globs, 1, -1)

    self.source_map = {}
    for s in source_infos:
      filename = os.path.basename(s.filepath)
      dupe = self.source_map.get(filename)
      if dupe:
        raise ValueError(
            'All base filenames must be unique. '
            f'Filename {filename} appears in both {s.filepath} and {dupe}.'
        )
      self.source_map[filename] = s.filepath


@dataclasses.dataclass
class BootstrapConfig:
  """Configuration for Search Bootstrap project."""

  # Embeddings dataset info.
  embeddings_glob: str
  embedding_hop_size_s: float
  audio_globs: Sequence[str] | None

  # Annotations info.
  # TODO(tomdenton): Write handling for the annotated data.
  annotated_path: str

  # Model info. Should match the model used for creating embeddings.
  model_key: str
  model_config: config_dict.ConfigDict
