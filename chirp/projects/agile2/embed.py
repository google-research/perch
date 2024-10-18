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

"""Functionality for embedding audio examples."""

import dataclasses
from typing import Iterator

from absl import logging
import audioread
from chirp import audio_utils
from chirp.projects.agile2 import source_info
from chirp.projects.hoplite import interface as hoplite_interface
from chirp.projects.zoo import model_configs
from chirp.projects.zoo import zoo_interface
from ml_collections import config_dict
import numpy as np
import soundfile


@dataclasses.dataclass
class ModelConfig(hoplite_interface.EmbeddingMetadata):
  """Configuration for embedding model.

  Attributes:
    model_key: Key for the model wrapper class.
    embedding_dim: Dimensionality of the embedding.
    model_config: Config dict of arguments to instantiate the model wrapper.
  """

  model_key: str
  embedding_dim: int
  model_config: config_dict.ConfigDict


class EmbedWorker:
  """Worker for embedding audio examples."""

  def __init__(
      self,
      audio_sources: source_info.AudioSources,
      model_config: ModelConfig,
      db: hoplite_interface.GraphSearchDBInterface,
      embedding_model: zoo_interface.EmbeddingModel | None = None,
  ):
    self.db = db
    self.model_config = model_config
    self.audio_sources = audio_sources
    if embedding_model is None:
      model_class = model_configs.MODEL_CLASS_MAP[model_config.model_key]
      self.embedding_model = model_class.from_config(model_config.model_config)
    else:
      self.embedding_model = embedding_model
    self.audio_globs = {
        g.dataset_name: g for g in self.audio_sources.audio_globs
    }

  def _log_error(self, source_id, exception, counter_name):
    logging.warning(
        'The audio at (%s / %f) could not be loaded (%s). '
        'The exception was (%s)',
        source_id.filepath,
        source_id.offset_s,
        counter_name,
        exception,
    )

  def _update_audio_sources(self):
    """Validates the embed config and/or saves it to the DB."""
    db_metadata = self.db.get_metadata(None)
    if 'audio_sources' not in db_metadata:
      self.db.insert_metadata(
          'audio_sources', self.audio_sources.to_config_dict()
      )
      return

    db_audio_sources = source_info.AudioSources.from_config_dict(
        db_metadata['audio_sources']
    )
    merged = self.audio_sources.merge_update(db_audio_sources)
    self.db.insert_metadata('audio_sources', merged.to_config_dict())
    self.audio_sources = merged

  def _update_model_config(self):
    """Validates the model config and/or saves it to the DB."""
    db_metadata = self.db.get_metadata(None)
    if 'model_config' not in db_metadata:
      self.db.insert_metadata(
          'model_config', self.model_config.to_config_dict()
      )
      return

    db_model_config = ModelConfig(**db_metadata['model_config'])
    if self.model_config == db_model_config:
      return

    # Validate the config against the DB.
    # TODO(tomdenton): Implement compatibility checks for model configs.
    if self.model_config.model_key != db_model_config.model_key:
      raise AssertionError(
          'The configured model key does not match the model key that is '
          'already in the DB.'
      )
    if self.model_config.embedding_dim != db_model_config.embedding_dim:
      raise AssertionError(
          'The configured embedding dimension does not match the embedding '
          'dimension that is already in the DB.'
      )
    self.db.insert_metadata('model_config', self.model_config.to_config_dict())

  def update_configs(self):
    """Validates the configs and saves them to the DB."""
    self._update_model_config()
    self._update_audio_sources()
    self.db.commit()

  def get_sample_rate_hz(self, source_id: source_info.SourceId) -> int:
    """Get the sample rate of the embedding model."""
    dataset_name = source_id.dataset_name
    if dataset_name not in self.audio_globs:
      raise ValueError(f'Dataset name {dataset_name} not found in audio globs.')
    audio_glob = self.audio_globs[dataset_name]
    if audio_glob.target_sample_rate_hz == -2:
      return self.embedding_model.sample_rate
    elif audio_glob.target_sample_rate_hz == -1:
      # Uses the file's native sample rate.
      return -1
    elif audio_glob.target_sample_rate_hz > 0:
      return audio_glob.target_sample_rate_hz
    else:
      raise ValueError('Invalid target_sample_rate.')

  def load_audio(self, source_id: source_info.SourceId) -> np.ndarray | None:
    """Load audio from the indicated source and log any problems."""
    try:
      audio_array = audio_utils.load_audio_window(
          source_id.filepath,
          source_id.offset_s,
          self.get_sample_rate_hz(source_id),
          source_id.shard_len_s,
      )
      return np.array(audio_array)
    except soundfile.LibsndfileError as inst:
      self._log_error(source_id, inst, 'audio_libsndfile_error')
    except ValueError as inst:
      self._log_error(source_id, inst, 'audio_bad_offset')
    except audioread.NoBackendError as inst:
      self._log_error(source_id, inst, 'audio_no_backend')
    except EOFError as inst:
      self._log_error(source_id, inst, 'audio_eof_error')
    except RuntimeError as inst:
      if 'Soundfile is not available' in str(inst):
        self._log_error(source_id, inst, 'audio_no_soundfile')
      else:
        self._log_error(source_id, inst, 'audio_runtime_error')

  def embedding_exists(self, source_id: source_info.SourceId) -> bool:
    """Check whether embeddings already exist for the given source ID."""
    embs = self.db.get_embeddings_by_source(
        dataset_name=source_id.dataset_name,
        source_id=source_id.file_id,
        offsets=np.array([source_id.offset_s], np.float16),
    )
    return embs.shape[0] > 0

  def process_source_id(
      self, source_id: source_info.SourceId
  ) -> Iterator[tuple[hoplite_interface.EmbeddingSource, np.ndarray]]:
    """Process a single audio source."""
    glob = self.audio_globs[source_id.dataset_name]
    audio_array = self.load_audio(source_id)
    if audio_array is None:
      return
    if (
        audio_array.shape[0]
        < glob.min_audio_len_s * self.embedding_model.sample_rate
    ):
      self._log_error(source_id, 'no_exception', 'audio_too_short')
      return

    if self.embedding_exists(source_id):
      self._log_error(source_id, 'no_exception', 'embeddings already exist')
      return

    outputs = self.embedding_model.embed(audio_array)
    embeddings = outputs.embeddings
    if embeddings is None:
      return
    hop_size_s = getattr(self.embedding_model, 'hop_size_s', 0.0)
    for t, embedding in enumerate(embeddings):
      offset_s = source_id.offset_s + t * hop_size_s
      emb_source_id = hoplite_interface.EmbeddingSource(
          dataset_name=source_id.dataset_name,
          source_id=source_id.file_id,
          offsets=np.array([offset_s], np.float16),
      )
      for channel_embedding in embedding:
        yield (emb_source_id, channel_embedding)

  def process_all(self, target_dataset_name: str | None = None):
    """Process all audio examples."""
    self.update_configs()
    # TODO(tomdenton): Prefetch audio in parallel for faster execution.
    for source_id in self.audio_sources.iterate_all_sources(
        target_dataset_name
    ):
      for emb_source_id, embedding in self.process_source_id(source_id):
        self.db.insert_embedding(embedding, emb_source_id)
    self.db.commit()
