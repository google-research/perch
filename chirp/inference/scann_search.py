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

"""Search through audio embedding using scann"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import chirp.inference.scann_search_lib as scann_lib

FLAGS = flags.FLAGS

_EMBEDDING_GLOB_PATH = flags.DEFINE_string(
    "embedding_glob_path",
    None,
    "The path to the embedding file directory",
)
_AUDIO_PATH = flags.DEFINE_string(
    "audio_path",
    "audio_path_placeholder",
    "The path to query audio file",
)
_EMBEDDING_MODEL_PATH = flags.DEFINE_string(
    "embedding_model_path",
    "embedding_model_path_placeholder",
    "The path to the embedding model",
)

_SCANN_OUTPUT_DIR = flags.DEFINE_string(
    "scann_output_dir",
    "output_dir_placeholder",
    "The path to the scann output directory",
)

flags.mark_flags_as_required([
    "embedding_glob_path",
    "audio_path",
    "embedding_model_path",
    "scann_output_dir",
])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print("creating searcher")
  searcher, embedding_files, timestamps = scann_lib.create_searcher(
      _EMBEDDING_GLOB_PATH.value, _SCANN_OUTPUT_DIR.value
  )

  print("embedding audio query")
  query_embedding = scann_lib.embed_query_audio(
      _AUDIO_PATH.value, _EMBEDDING_MODEL_PATH.value
  )

  print("searching query")
  results = scann_lib.search_query(
      searcher, query_embedding, embedding_files, timestamps
  )


if __name__ == "__main__":
  app.run(main)
