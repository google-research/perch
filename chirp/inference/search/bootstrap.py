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

"""Configuration and init library for Search Bootstrap projects."""

import dataclasses
import functools
import hashlib
from typing import Callable, Iterator, Sequence

from chirp import audio_utils
from chirp.inference import baw_utils
from chirp.inference import embed_lib
from chirp.inference import interface
from chirp.inference import models
from chirp.inference import tf_examples
from chirp.inference.search import search
from etils import epath
from ml_collections import config_dict
import tensorflow as tf


@dataclasses.dataclass
class BootstrapState:
  """Union of data and models useful to go from a few examples to a detector.

  Attributes:
    config: The configuration of the bootstrap project.
    embedding_model: The model used to compute embeddings, loaded on init.
    embeddings_dataset: A TF Dataset of the embeddings, loaded on init.
    source_map: A Callable mapping file_id to full filepath.
    baw_auth_token: Auth token for fetching BAW/A2O data.
  """

  config: 'BootstrapConfig'
  embedding_model: interface.EmbeddingModel | None = None
  embeddings_dataset: tf.data.Dataset | None = None
  source_map: Callable[[str, float], str] | None = None
  baw_auth_token: str = ''

  def __post_init__(self):
    if self.embedding_model is None:
      self.embedding_model = models.model_class_map()[
          self.config.model_key
      ].from_config(self.config.model_config)
    self.create_embeddings_dataset()
    if self.source_map is None:
      if self.baw_auth_token:
        window_size_s = self.config.model_config.window_size_s
        self.source_map = functools.partial(
            baw_utils.make_baw_audio_url_from_file_id,
            window_size_s=window_size_s,
        )
      else:
        self.source_map = lambda file_id, offset: filesystem_source_map(
            self.config.audio_globs, self.config.file_id_depth, file_id
        )

  def create_embeddings_dataset(self, shuffle_files: bool = False):
    """Create a TF Dataset of the embeddings."""
    if self.embeddings_dataset and not shuffle_files:
      return self.embeddings_dataset
    ds = tf_examples.create_embeddings_dataset(
        self.config.embeddings_path,
        self.config.embeddings_glob,
        tensor_dtype=self.config.tensor_dtype,
        shuffle_files=shuffle_files,
    )
    self.embeddings_dataset = ds
    return ds

  def search_results_audio_iterator(
      self, search_results: search.TopKSearchResults, **kwargs
  ) -> Iterator[search.SearchResult]:
    """Create an iterator over TopKSearchResults which loads audio."""
    filepaths = [
        self.source_map(r.filename, r.timestamp_offset)
        for r in search_results.search_results
    ]
    offsets = [r.timestamp_offset for r in search_results.search_results]
    sample_rate = self.config.model_config.sample_rate
    window_size_s = self.config.model_config.window_size_s
    if self.baw_auth_token:
      iterator = baw_utils.multi_load_baw_audio(
          filepaths=filepaths,
          offsets=offsets,
          auth_token=self.baw_auth_token,
          sample_rate=sample_rate,
          **kwargs,
      )
    else:
      audio_loader = functools.partial(
          audio_utils.load_audio_window,
          window_size_s=window_size_s,
          sample_rate=sample_rate,
      )
      iterator = audio_utils.multi_load_audio_window(
          filepaths=filepaths,
          offsets=offsets,
          audio_loader=audio_loader,
          **kwargs,
      )
    for result, audio in zip(search_results.search_results, iterator):
      result.audio = audio
      yield result


@dataclasses.dataclass
class BootstrapConfig:
  """Configuration for Search Bootstrap project."""

  # Embeddings dataset info.
  embeddings_path: str

  # Annotations info.
  annotated_path: str

  # Tensor dtype in embeddings.
  tensor_dtype: str

  # Glob for embeddings.
  embeddings_glob: str

  # The following are populated automatically from the embedding config.
  embedding_hop_size_s: float
  file_id_depth: int
  audio_globs: Sequence[str]
  model_key: str
  model_config: config_dict.ConfigDict
  tf_record_shards: int

  @classmethod
  def load_from_embedding_path(cls, embeddings_path: str, **kwargs):
    """Instantiate from a configuration written alongside embeddings."""
    embedding_config = embed_lib.load_embedding_config(embeddings_path)
    return cls.load_from_embedding_config(
        embedding_config, embeddings_path=embeddings_path, **kwargs
    )

  @classmethod
  def load_from_embedding_config(
      cls,
      embedding_config: config_dict.ConfigDict,
      annotated_path: str,
      tf_record_shards: int = 1,
      embeddings_path: str | None = None,
      embeddings_glob: str = 'embeddings-*',
  ):
    """Instantiate from an embedding config."""
    embed_fn_config = embedding_config.embed_fn_config
    tensor_dtype = embed_fn_config.get('tensor_dtype', 'float32')
    tf_record_shards = embedding_config.get(
        'tf_record_shards', tf_record_shards
    )

    # Extract the embedding model config from the embedding_config.
    if embed_fn_config.model_key == 'separate_embed_model':
      # If a separation model was applied, get the embedding model config only.
      model_key = 'taxonomy_model_tf'
      model_config = embed_fn_config.model_config.taxonomy_model_tf_config
    else:
      model_key = embed_fn_config.model_key
      model_config = embed_fn_config.model_config
    if embeddings_path is None:
      embeddings_path = embedding_config.output_dir
    return BootstrapConfig(
        embeddings_path=embeddings_path,
        annotated_path=annotated_path,
        model_key=model_key,
        model_config=model_config,
        embedding_hop_size_s=model_config.hop_size_s,
        file_id_depth=embed_fn_config.file_id_depth,
        audio_globs=embedding_config.source_file_patterns,
        tensor_dtype=tensor_dtype,
        tf_record_shards=tf_record_shards,
        embeddings_glob=embeddings_glob,
    )

  def embedding_config_hash(self, digest_size: int = 10) -> str:
    """Returns a stable hash of the model key and config."""
    config_str = self.model_config.to_json(sort_keys=True)
    encoded_str = f'{self.model_key};{config_str}'.encode('utf-8')

    hash_obj = hashlib.blake2b(digest_size=digest_size)
    hash_obj.update(encoded_str)
    return hash_obj.hexdigest()


def filesystem_source_map(
    audio_globs: Sequence[str],
    file_id_depth: int,
    file_id: str,
) -> str:
  """Map filenames to full filepaths."""
  if audio_globs is None:
    raise ValueError('No audio globs found in the embedding config.')

  for path_glob in audio_globs:
    # First check for the file using the (known) file_id_depth.
    base_path = epath.Path(path_glob).parts[: -file_id_depth - 1]
    candidate_path = epath.Path('').joinpath(*base_path) / file_id
    if candidate_path.exists():
      return candidate_path.as_posix()

    # Remove any wildcards from the path, and append the file_id.
    # This assumes that wildcards are only used at the end of the path,
    # but this asusmption is not enforced.
    base_path = '/'.join([p for p in path_glob.split('/') if '*' not in p])
    candidate_path = epath.Path(base_path) / file_id
    if candidate_path.exists():
      return candidate_path.as_posix()
  raise ValueError(f'No file found for file_id {file_id}.')
