# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Embed audio data with BirdNET."""

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

  # Note that the model path should be either the location of the '.tflite'
  # file or the directory contraining the 'saved_model.pb'.
  model_path = ''

  config.num_shards_per_file = 1
  config.shard_len_s = 60
  # Number of workers when using the Beam DirectRunner on a single machine.
  config.num_direct_workers = 8
  config.embed_fn_config = {
      'write_embeddings': True,
      'write_logits': True,
      'write_separated_audio': False,
      'write_raw_audio': True,
      'model_key': 'birdnet',
      'model_config': {
          'model_path': model_path,
          'window_size_s': 3.0,
          'hop_size_s': 3.0,
          'sample_rate': 48000,
      },
  }
  return config
