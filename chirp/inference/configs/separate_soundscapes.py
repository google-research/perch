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

"""Embed audio data using both a seapration and embedding model."""

from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config
_object_config = config_utils.object_config


def get_config() -> config_dict.ConfigDict:
  """Create the Caples inference config."""
  # Attention-based 5s model.
  config = config_dict.ConfigDict()

  config.output_dir = ''
  config.source_file_patterns = []
  sep_model_checkpoint_path = ''
  emb_model_checkpoint_path = ''

  config.shard_len_s = -1
  config.num_shards_per_file = -1

  # Number of workers when using the Beam DirectRunner on a single machine.
  config.num_direct_workers = 8
  config.embed_fn_config = {
      'write_embeddings': True,
      'write_logits': False,
      'write_separated_audio': False,
      'write_raw_audio': False,
      'file_id_depth': 1,
      'model_key': 'separate_embed_model',
      'file_id_depth': 1,
      'model_config': {
          'sample_rate': 32000,
          'taxonomy_model_tf_config': {
              'model_path': emb_model_checkpoint_path,
              'window_size_s': 5.0,
              'hop_size_s': 5.0,
              'sample_rate': 32000,
          },
          'separator_model_tf_config': {
              'model_path': sep_model_checkpoint_path,
              'sample_rate': 32000,
              'frame_size': 32000,
          },
      },
  }
  return config
