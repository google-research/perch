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

"""Create embeddings for an audio corpus."""

import dataclasses
import os
from typing import Any, Dict, Optional, Sequence
import warnings

from absl import logging
import apache_beam as beam
from chirp.inference import models
import librosa
from ml_collections import config_dict
import numpy as np
import tensorflow as tf

FILE_NAME = 'filename'
TIMESTAMP = 'timestamp_offset'
EMBEDDING = 'embedding'
EMBEDDING_SHAPE = 'embedding_shape'
LOGITS = 'logits'
SEPARATED_AUDIO = 'separated_audio'
SEPARATED_AUDIO_SHAPE = 'separated_audio_shape'

MODEL_CLASSES: Dict[str, Any] = {
    'taxonomy_model_tf': models.TaxonomyModelTF,
    'birdnet': models.BirdNet,
    'dummy_model': models.DummyModel,
}


@dataclasses.dataclass
class SourceInfo:
  """Source information for extracting target audio from a file."""
  filepath: str
  shard_num: int
  num_shards: int


def get_feature_description(logit_names: Optional[Sequence[str]] = None):
  """Create a feature description for the TFExamples."""
  feature_description = {
      FILE_NAME:
          tf.io.FixedLenFeature([], tf.string),
      TIMESTAMP:
          tf.io.FixedLenFeature([], tf.int64),
      EMBEDDING:
          tf.io.FixedLenFeature([], tf.string),
      SEPARATED_AUDIO:
          tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
  }
  if logit_names is not None:
    for logit_name in logit_names:
      feature_description[logit_name] = tf.io.FixedLenFeature([], tf.string)
      feature_description[logit_name + '_shape'] = tf.io.FixedLenFeature(
          [], tf.int64)
  return feature_description


def create_source_infos(source_files, num_shards_per_file):
  source_file_splits = []
  for source in source_files:
    for i in range(num_shards_per_file):
      source_file_splits.append(SourceInfo(source, i, num_shards_per_file))
  return source_file_splits


class EmbedFn(beam.DoFn):
  """Beam worker function for creating audio embeddings."""

  def __init__(self, hop_size_s: float, write_embeddings: bool,
               write_logits: bool, write_separated_audio: bool, model_key: str,
               model_config: config_dict.ConfigDict):
    """Initialize the embedding DoFn.

    Args:
      hop_size_s: Number of seconds to hop. Ignored if the model handles
        windowing itself; ie, the model's window_size_s == -1.
      write_embeddings: Whether to write embeddings.
      write_logits: Whether to write output logits.
      write_separated_audio: Whether to write out separated audio tracks.
      model_key: String indicating which model wrapper to use. See MODEL_KEYS.
      model_config: Keyword arg dictionary for the model wrapper class.
    """

    self.hop_size_s = hop_size_s
    self.model_key = model_key
    self.model_config = model_config
    self.write_embeddings = write_embeddings
    self.write_logits = write_logits
    self.write_separated_audio = write_separated_audio

  def setup(self):
    self.embedding_model = MODEL_CLASSES[self.model_key](**self.model_config)

  def load_audio(self, filepath: str) -> Optional[np.ndarray]:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      try:
        audio, _ = librosa.load(
            filepath, sr=self.embedding_model.sample_rate, res_type='polyphase')
      except Exception as inst:  # pylint: disable=broad-except
        # We have no idea what can go wrong in librosa, so we catch a broad
        # exception here.
        logging.warning(
            'The audio at %s could not be loaded. '
            'The exception was (%s)', filepath, inst)
        return None
      while len(audio.shape) > 1:
        # In case of multi-channel audio, take the first channel.
        audio = audio[0]
      return audio

  def maybe_frame_audio(self, audio: np.ndarray) -> np.ndarray:
    window_size_s = self.embedding_model.window_size_s
    if window_size_s < 0:
      return audio
    frame_length = int(window_size_s * self.embedding_model.sample_rate)
    hop_length = int(self.hop_size_s * self.embedding_model.sample_rate)
    # Librosa frames as [frame_length, batch], so need a transpose.
    framed_audio = librosa.util.frame(audio, frame_length, hop_length).T
    return framed_audio

  def serialize_tensor(self, tensor: tf.Tensor) -> np.ndarray:
    serialized = tf.io.serialize_tensor(tensor)
    return serialized.numpy()

  def embed(self, file_id: str, audio: np.ndarray,
            timestamp_offset: int) -> tf.train.Example:
    """Apply the embedding model to the target audio array."""
    framed_audio = self.maybe_frame_audio(audio)
    logging.info('...creating embeddings (%s)', file_id)
    outputs = self.embedding_model.embed(framed_audio)
    if self.embedding_model.window_size_s < 0:
      # In this case, the model determines its own hop size.
      hops_size_samples = audio.shape[0] // outputs.embeddings.shape[0]
    else:
      # In this case, we use the config hop size.
      hops_size_samples = int(self.hop_size_s *
                              self.embedding_model.sample_rate)
    for i in range(outputs.embeddings.shape[0]):
      offset = timestamp_offset + i * hops_size_samples
      feature = {
          FILE_NAME: bytes_feature(bytes(file_id, encoding='utf8')),
          TIMESTAMP: int_feature(offset),
      }
      if self.write_embeddings and outputs.embeddings is not None:
        feature[EMBEDDING] = bytes_feature(
            self.serialize_tensor(outputs.embeddings[i]))
        feature[EMBEDDING_SHAPE] = int_feature(outputs.embeddings[i].shape),
      if self.write_logits and outputs.logits is not None:
        for logits_key, value in outputs.logits.items():
          feature[logits_key] = bytes_feature(self.serialize_tensor(value[i]))
          feature[logits_key + '_shape'] = int_feature(value[i].shape)
      if self.write_separated_audio and outputs.separated_audio is not None:
        feature[SEPARATED_AUDIO] = bytes_feature(
            self.serialize_tensor(outputs.separated_audio[i]))
        feature[SEPARATED_AUDIO_SHAPE] = int_feature(
            outputs.separated_audio[i].shape)
      ex = tf.train.Example(features=tf.train.Features(feature=feature))
      beam.metrics.Metrics.counter('beaminference', 'examples_processed').inc()
      yield ex.SerializeToString()
    beam.metrics.Metrics.counter('beaminference', 'segments_processed').inc()

  @beam.typehints.with_output_types(Any)
  def process(self, source_info: SourceInfo, crop_s=-1):
    """Process a source.

    Args:
      source_info: SourceInfo describing the audio to process.
      crop_s: If >0, only the first crop_s seconds will be used. Helpful for
        dry-run testing.

    Yields:
      A TFExample.
    """
    file_name = os.path.basename(source_info.filepath)

    logging.info('...loading audio (%s)', source_info.filepath)
    audio = self.load_audio(source_info.filepath)
    if audio is None:
      beam.metrics.Metrics.counter('beaminference', 'load_audio_error').inc()
      logging.error('Failed to load audio : %s', source_info.filepath)
      return

    if audio.shape[0] < 5 * self.embedding_model.sample_rate:
      beam.metrics.Metrics.counter('beaminference', 'short_audio_error').inc()
      logging.error('short audio file : %s', source_info.filepath)
      return

    if source_info.num_shards > 1:
      shard_len = audio.shape[0] // source_info.num_shards
      timestamp_offset = source_info.shard_num * shard_len
      audio = audio[timestamp_offset:timestamp_offset + shard_len]
    else:
      timestamp_offset = 0

    if crop_s > 0:
      audio = audio[:crop_s * self.embedding_model.sample_rate]
    for example in self.embed(file_name, audio, timestamp_offset):
      yield example


def bytes_feature(x, default=''):
  if x is None:
    x = default
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def int_feature(x, default=-1):
  if x is None:
    x = default
  if hasattr(x, 'count'):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))
