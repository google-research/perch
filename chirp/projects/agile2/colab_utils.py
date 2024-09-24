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

"""Utility functions for sqlite-backed Agile modeling notebooks."""

import dataclasses

from chirp.projects.agile2 import embed
from chirp.projects.hoplite import db_loader
from chirp.projects.hoplite import interface
from chirp.projects.zoo import models
from etils import epath
from ml_collections import config_dict


@dataclasses.dataclass
class AgileConfigs:
  """Container for the various configs used in the Agile notebooks."""

  # Config for the raw audio sources.
  audio_sources_config: embed.EmbedConfig
  # Database config for the embeddings database.
  db_config: db_loader.DBConfig
  # Config for the embedding model.
  model_config: embed.ModelConfig

  def as_config_dict(self) -> config_dict.ConfigDict:
    """Returns the configs as a ConfigDict."""
    return config_dict.ConfigDict({
        'audio_sources_config': self.audio_sources_config.to_config_dict(),
        'db_config': self.db_config.to_config_dict(),
        'model_config': self.model_config.to_config_dict(),
    })


def validate_and_save_configs(
    configs: AgileConfigs,
    db: interface.GraphSearchDBInterface,
):
  """Validates that the model config is compatible with the DB."""

  model_config = configs.model_config
  db_metadata = db.get_metadata(None)
  if 'model_config' in db_metadata:
    if db_metadata['model_config'].model_key != model_config.model_key:
      raise AssertionError(
          'The configured embedding model does not match the embedding model'
          ' that is already in the DB.  You either need to drop the database or'
          " use the '%s' model confg."
          % db_metadata['model_config'].model_key
      )

  db.insert_metadata('model_config', model_config.to_config_dict())
  db.insert_metadata(
      'embed_config', configs.audio_sources_config.to_config_dict()
  )
  db.commit()


def load_configs(
    audio_globs: dict[str, tuple[str, str]],
    db_path: str,
    model_config_key: str = 'perch_8',
) -> AgileConfigs:
  """Load default configs for the notebook and return them as an AgileConfigs.

  Args:
    audio_globs: Mapping from dataset name to pairs of `(root directory, file
      glob)`.
    db_path: Location of the database.  If None, the database will be created in
      the same directory as the audio.
    model_config_key: Name of the embedding model to use.

  Returns:
    AgileConfigs object with the loaded configs.
  """
  if db_path is None:
    if len(audio_globs) > 1:
      raise ValueError(
          'db_path must be specified when embedding multiple datasets.'
      )
    # Put the DB in the same directory as the audio.
    db_path = (
        epath.Path(next(iter(audio_globs.values()))[0]) / 'hoplite_db.sqlite'
    )

  model_key, embedding_dim, model_config = models.get_preset_model_config(
      model_config_key
  )
  db_model_config = embed.ModelConfig(
      model_key=model_key,
      embedding_dim=embedding_dim,
      model_config=model_config,
  )
  db_config = config_dict.ConfigDict({
      'db_path': db_path,
      'embedding_dim': embedding_dim,
  })

  audio_srcs_config = embed.EmbedConfig(
      audio_globs=audio_globs,
      min_audio_len_s=1.0,
  )

  return AgileConfigs(
      audio_sources_config=audio_srcs_config,
      db_config=db_loader.DBConfig('sqlite', db_config),
      model_config=db_model_config,
  )
