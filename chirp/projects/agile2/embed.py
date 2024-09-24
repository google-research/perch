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
from chirp.projects.zoo import models
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


@dataclasses.dataclass
class EmbedConfig(hoplite_interface.EmbeddingMetadata):
  """Configuration for embedding processing.

  Attributes:
    audio_globs: Mapping from dataset name to pairs of `(root directory, file
      glob)`.
    min_audio_len_s: Minimum audio length to process.
    target_sample_rate_hz: Target sample rate for audio. If -2, use the
      embedding model's declared sample rate. If -1, use the file's native
      sample rate. If > 0, resample to the specified rate.
  """

  audio_globs: dict[str, tuple[str, str]]
  min_audio_len_s: float
  target_sample_rate_hz: int = -1


class EmbedWorker:
  """Worker for embedding audio examples."""

  def __init__(
      self,
      embed_config: EmbedConfig,
      model_config: ModelConfig,
      db: hoplite_interface.GraphSearchDBInterface,
      embedding_model: zoo_interface.EmbeddingModel | None = None,
  ):
    self.db = db
    self.model_config = model_config
    self.embed_config = embed_config
    if embedding_model is None:
      model_class = models.model_class_map()[model_config.model_key]
      self.embedding_model = model_class.from_config(model_config.model_config)
    else:
      self.embedding_model = embedding_model

  def _log_error(self, source_id, exception, counter_name):
    logging.warning(
        'The audio at (%s / %f) could not be loaded (%s). '
        'The exception was (%s)',
        source_id.filepath,
        source_id.offset_s,
        counter_name,
        exception,
    )

  def get_sample_rate_hz(self) -> int:
    """Get the sample rate of the embedding model."""
    if self.embed_config.target_sample_rate_hz == -2:
      return self.embedding_model.sample_rate
    elif self.embed_config.target_sample_rate_hz == -1:
      # Uses the file's native sample rate.
      return -1
    elif self.embed_config.target_sample_rate_hz > 0:
      return self.embed_config.target_sample_rate_hz
    else:
      raise ValueError('Invalid target_sample_rate.')

  def load_audio(self, source_id: source_info.SourceId) -> np.ndarray | None:
    """Load audio from the indicated source and log any problems."""
    try:
      audio_array = audio_utils.load_audio_window(
          source_id.filepath,
          source_id.offset_s,
          self.embed_config.target_sample_rate_hz,
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
    audio_array = self.load_audio(source_id)
    if audio_array is None:
      return
    if (
        audio_array.shape[0]
        < self.embed_config.min_audio_len_s * self.embedding_model.sample_rate
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

  def process_all(self):
    """Process all audio examples."""
    audio_sources = source_info.AudioSources(self.embed_config.audio_globs)
    for source_id in audio_sources.iterate_all_sources():
      for emb_source_id, embedding in self.process_source_id(source_id):
        self.db.insert_embedding(embedding, emb_source_id)
    self.db.commit()
