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

"""Utility functions for working with the A2O API."""

import os
from ml_collections import config_dict

def get_a2o_embeddings_config() -> config_dict.ConfigDict:
  """Returns an embeddings config for the public A2O embeddings."""
  chirp_public_bucket = "gs://chirp-public-bucket"
  perch_512_model_path = os.path.join(chirp_public_bucket, "models/perch_4_512")
  embeddings_uri = os.path.join(
      chirp_public_bucket, "embeddings/a2o_embeddings_perch512"
  )

  config = config_dict.ConfigDict({
      "output_dir": embeddings_uri,
      "source_file_patterns": [
          "https://api.acousticobservatory.org/audio_recordings/download/flac/*",
      ],
      "num_shards_per_file": 1,
      "shard_len_s": 60,
      "start_shard_idx": 0,
      "num_direct_workers": 8,
      "embed_fn_config": {
          "write_embeddings": True,
          "write_logits": False,
          "write_separated_audio": False,
          "write_raw_audio": False,
          "file_id_depth": 1,
          "model_key": "taxonomy_model_tf",
          "tensor_dtype": "float16",
          "model_config": {
              "model_path": perch_512_model_path,
              "window_size_s": 5.0,
              "hop_size_s": 5.0,
              "sample_rate": 32000,
          },
          "logits_head_config": {
              "model_path": perch_512_model_path + "/speech_empty_filter",
              "logits_key": "nuisances",
              "channel_pooling": "",
          },
      },
  })
  return config
