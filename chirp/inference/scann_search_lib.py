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

"""Utility functions for scann search"""

import dataclasses
import math
import os
import sys
import time

from absl import logging
from chirp import audio_utils
from chirp.inference import models
from chirp.inference.embed_lib import load_embedding_config
from chirp.inference.search import bootstrap
from chirp.inference.tf_examples import get_example_parser
from etils import epath
from ml_collections import config_dict
import numpy as np
import tensorflow as tf

from scann.scam_ops.py import scam_ops_pybind


@dataclasses.dataclass(frozen=True)
class AudioSearchResult:
  """Attributes:

  index: Index for the searcher ndarray.
  distance: The nearest neighbor distance calculated by scann searcher.
  filename: The filename of the source audio file.
  timestamp_offset_s: Timestamp offset in seconds for the audio file.
  """

  index: int
  distance: float
  filename: str
  timestamp_offset_s: float


# TODO(joycehsy): handle embedding shape automatically.
def create_searcher(
    embeddings_glob: str,
    output_dir: str,
    num_neighbors: int = 10,
    embedding_shape: tuple = (12, 1, 1280),
    distance_measure: str = "squared_l2",
    embedding_list_filename="embedding_list.txt",
    timestamps_list_filename="timestamps_list.txt",
) -> tuple[scam_ops_pybind.ScamSearcher, list[str], list[int]]:
  """Creates Scann searcher from all embedding files within given directory.

  In order to index back to the original audio chunk using the Scann returned
  search index, we have to save also the list of embedding filenames and their
  timestamps. This information is found the embedding_list_filename and
  timestamps_list_filename.

  We checked whether the embedding chunk has the shape of (12, 1, 1280), as the
  shape can be slightly shorter because of the remainder chunk when dividing.

  Args:
    embedding_glob: Path the directory containing audio embeddings produced by
      the embedding model that matches the embedding_shape.
    output_dir: Output directory path to save the scann artifacts.
    num_neighbors: Number of neighbors for scann search.
    embedding_shape: Hidden dim from EmbeddingModel.
    distance_measure: One of "squared_l2" or "dot_product".
    embedding_list_filename: output filename for generated list of embeddings.
    timestamps_list_filename: output filename for generated list of timestamps.

  Returns:
    The searcher object for for Scann, file with embedding_list_filename, file
    with timestamps_list_filename.
  """
  if (
      os.path.exists(output_dir)
      and os.path.exists(os.path.join(output_dir, embedding_list_filename))
      and os.path.exists(os.path.join(output_dir, timestamps_list_filename))
  ):
    searcher = scam_ops_pybind.load_searcher(output_dir)
    filenames = np.loadtxt(
        os.path.join(output_dir, embedding_list_filename)
    ).tolist()
    timestamps = np.loadtxt(
        os.path.join(output_dir, timestamps_list_filename)
    ).tolist()
    if len(filenames) != len(timestamps):
      raise ValueError(
          "Error loading filenames and timestamps. The two lists length are not"
          " equal but should be."
      )
    return searcher, filenames, timestamps

  if distance_measure != "squared_l2" and distance_measure != "dot_product":
    raise ValueError("Distance measure must be squared_l2 or dot_product.")

  embeddings_glob = epath.Path(embeddings_glob)
  embeddings_files = [
      fn.as_posix() for fn in embeddings_glob.glob("embeddings-002*")
  ]
  ds = tf.data.TFRecordDataset(
      embeddings_files, num_parallel_reads=tf.data.AUTOTUNE
  )
  parser = get_example_parser()
  ds = ds.map(parser)

  embedding_config = load_embedding_config(embeddings_glob)
  hop_size_s = embedding_config.embed_fn_config.model_config.hop_size_s

  # These will be saved to output files.
  embeddings = []
  timestamps_output = []
  filenames_output = []

  # Drop remainder of embedding if first dimension is < 12.
  # todo(joycehsy): fix to pad zeros for dimensions < 12.
  for example in ds:
    if example["embedding"].shape == embedding_shape:
      embeddings.append(example["embedding"].numpy())

      if example["embedding"].shape[1] != 1:
        raise NotImplementedError("channel size is non-trivial != 1.")

      for i in range(int(example["embedding"].shape[0])):
        filenames_output.append(example["filename"])
        timestamps_output.append(example["timestamp_s"] + i * hop_size_s)

  embeddings = np.array(embeddings)
  embeddings = embeddings.reshape(-1, embeddings.shape[-1])

  logging.info(
      "Embedding arrays shape: %s, used to build scann searcher",
      embeddings.shape,
  )

  searcher = (
      scam_ops_pybind.builder(embeddings, num_neighbors, distance_measure)
      .score_brute_force(False)
      .build()
  )

  # Saving objects to output directory.
  os.makedirs(os.path.dirname(output_dir), exist_ok=True)
  searcher.serialize(output_dir)
  np.savetxt(
      os.path.join(output_dir, embedding_list_filename),
      np.array(filenames_output),
      fmt="%s",
  )
  np.savetxt(
      os.path.join(output_dir, timestamps_list_filename),
      np.array(timestamps_output),
      fmt="%d",
  )

  return (searcher, filenames_output, timestamps_output)


def embed_query_audio(
    audio_path: str,
    embedding_model_path: str,
    sample_rate: int = 32000,
    window_size_s: float = 5.0,
    hop_size_s: float = 5.0,
    embedding_hidden_dims: int = 1280,
) -> np.ndarray:
  """Embeds the audio query through embedding the model.

  Args:
    audio_path: File path to audio query.
    embedding_model_path: File path to saved embedding model.
    sample_rate: Sampling rate for the model.
    window_size_s: Window size of the model in seconds.
    hop_size_s: Hop size for processing longer audio files.
    embedding_hidden_dims: Embedding model's hidden dimension size.

  Returns:
    Query audio embedding as numpy array.
  """
  query_audio = audio_utils.load_audio(audio_path, sample_rate)
  logging.info("Query audio shape: %s", query_audio.shape)

  config = config_dict.ConfigDict({
      "model_path": embedding_model_path,
      "sample_rate": sample_rate,
      "window_size_s": window_size_s,
      "hop_size_s": hop_size_s,
  })
  embedding_model = models.TaxonomyModelTF.from_config(config)

  outputs = embedding_model.embed(np.array(query_audio))

  query_embedding = outputs.pooled_embeddings("first", "first").reshape(-1, 1)

  logging.info("Query after embeddding shape: %s", query_embedding.shape)
  return query_embedding.T


def search_query(
    searcher: scam_ops_pybind.ScamSearcher,
    query_embedding: np.ndarray,
    embedding_files: list[str],
    timestamps: list[int],
) -> list[AudioSearchResult]:
  """Searches for the neighbors of the query embedding using scann searcher.

  Args:
    searcher: The Scann searcher object.
    query_embedding: Embedding of the audio query.
    embedding_files: List of embedding files loaded from the generated output
      "embedding_list_filename" from create_searcher.
    timestamps: List of timestamps loaded from the generated output
      "embedding_list_filename" from create_searcher.

  Returns:
    A list of AudioSearchResults.
  """
  indices, distances = searcher.search_batched(query_embedding)
  results = []
  for idx, dis in zip(indices[0], distances[0]):
    r = AudioSearchResult(idx, dis, embedding_files[idx], timestamps[idx])
    results.append(r)

  return results
