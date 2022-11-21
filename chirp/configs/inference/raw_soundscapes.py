# coding=utf-8
# Copyright 2022 The Chirp Authors.
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

"""Embed Caples data."""

from chirp import config_utils
from ml_collections import config_dict

_c = config_utils.callable_config
_object_config = config_utils.object_config


def get_config() -> config_dict.ConfigDict:
  """Create the Caples inference config."""
  # Attention-based 5s model.
  config = config_dict.ConfigDict()

  config.output_dir = ''
  config.source_file_patterns = ['soundscapes/*.wav']
  model_checkpoint_path = ''

  config.num_shards_per_file = 1
  config.embed_fn_config = {
      'hop_size_s': 2.5,
      'model_key': 'taxonomy_model_tf',
      'model_config': {
          'model_path': model_checkpoint_path,
          'window_size_s': 5,
          'sample_rate': 32000,
      },
  }
  return config
