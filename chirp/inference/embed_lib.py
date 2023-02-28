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
from typing import Any, Sequence
import warnings

from absl import logging
import apache_beam as beam
from chirp.inference import interface
from chirp.inference import models
from etils import epath
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
RAW_AUDIO = 'raw_audio'
RAW_AUDIO_SHAPE = 'raw_audio_shape'
MIN_AUDIO_S = 5

MODEL_CLASSES: dict[str, Any] = {
    'taxonomy_model_tf': models.TaxonomyModelTF,
    'separator_model_tf': models.SeparatorModelTF,
    'birdnet': models.BirdNet,
    'placeholder_model': models.PlaceholderModel,
}


@dataclasses.dataclass
class SourceInfo:
  """Source information for extracting target audio from a file."""

  filepath: str
  shard_num: int
  num_shards: int


def get_feature_description(logit_names: Sequence[str] | None = None):
  """Create a feature description for the TFExamples.

  Each tensor feature includes both a serialized tensor and a 'shape' feature.
  The tensor feature can be parsed with tf.io.parse_tensor, and then reshaped
  according to the shape feature.

  Args:
    logit_names: Name of logit features included in the examples.

  Returns:
    Feature description dict for parsing TF Example protos.
  """
  feature_description = {
      FILE_NAME: tf.io.FixedLenFeature([], tf.string),
      TIMESTAMP: tf.io.FixedLenFeature([], tf.int64),
      EMBEDDING: tf.io.FixedLenFeature([], tf.string, default_value=''),
      EMBEDDING_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      SEPARATED_AUDIO: tf.io.FixedLenFeature([], tf.string, default_value=''),
      SEPARATED_AUDIO_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      RAW_AUDIO: tf.io.FixedLenFeature([], tf.string, default_value=''),
      RAW_AUDIO_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
  }
  if logit_names is not None:
    for logit_name in logit_names:
      feature_description[logit_name] = tf.io.FixedLenFeature(
          [], tf.string, default_value=''
      )
      feature_description[f'{logit_name}_shape'] = (
          tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
      )
  return feature_description


def create_source_infos(
    source_file_patterns: str, num_shards_per_file: int
) -> Sequence[SourceInfo]:
  """Expand source file patterns into a list of SourceInfos."""
  source_files = []
  for pattern in source_file_patterns:
    for source_file in epath.Path('').glob(pattern):
      source_files.append(source_file)

  source_file_splits = []
  for source in source_files:
    for i in range(num_shards_per_file):
      source_file_splits.append(
          SourceInfo(source.as_posix(), i, num_shards_per_file)
      )
  return source_file_splits


class EmbedFn(beam.DoFn):
  """Beam worker function for creating audio embeddings.

  TODO(tomdenton): Move most of this functionality into the EmbeddingModel.
  This will increase usability in non-beam contexts.
  """

  def __init__(
      self,
      write_embeddings: bool,
      write_logits: bool,
      write_separated_audio: bool,
      write_raw_audio: bool,
      model_key: str,
      model_config: config_dict.ConfigDict,
      crop_s: float = -1.0,
      embedding_model: interface.EmbeddingModel | None = None,
  ):
    """Initialize the embedding DoFn.

    Args:
      write_embeddings: Whether to write embeddings.
      write_logits: Whether to write output logits.
      write_separated_audio: Whether to write out separated audio tracks.
      write_raw_audio: If true, will add the original audio to the output.
      model_key: String indicating which model wrapper to use. See MODEL_KEYS.
        Only used for setting up the embedding model.
      model_config: Keyword arg dictionary for the model wrapper class. Only
        used for setting up the embedding model.
      crop_s: If greater than zero, run on only the first crop_s seconds.
      embedding_model: Pre-loaded embedding model.
    """
    self.model_key = model_key
    self.model_config = model_config
    self.write_embeddings = write_embeddings
    self.write_logits = write_logits
    self.write_separated_audio = write_separated_audio
    self.write_raw_audio = write_raw_audio
    self.crop_s = crop_s
    self.embedding_model = embedding_model

  def setup(self):
    if self.embedding_model is None:
      self.embedding_model = MODEL_CLASSES[self.model_key](**self.model_config)
    if hasattr(self, 'model_key'):
      del self.model_key
    if hasattr(self, 'model_config'):
      del self.model_config

  def load_audio(self, filepath: str) -> np.ndarray | None:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      try:
        audio, _ = librosa.load(
            filepath, sr=self.embedding_model.sample_rate, res_type='polyphase'
        )
      except Exception as inst:  # pylint: disable=broad-except
        # We have no idea what can go wrong in librosa, so we catch a broad
        # exception here.
        logging.warning(
            'The audio at %s could not be loaded. The exception was (%s)',
            filepath,
            inst,
        )
        return None
      while len(audio.shape) > 1:
        # In case of multi-channel audio, take the first channel.
        audio = audio[0]
      return audio

  def maybe_frame_audio(self, audio: np.ndarray) -> np.ndarray:
    if hasattr(self.embedding_model, 'window_size_s'):
      window_size_s = self.embedding_model.window_size_s
    else:
      return audio

    if window_size_s <= 0:
      return audio
    frame_length = int(window_size_s * self.embedding_model.sample_rate)
    hop_length = int(self.hop_size_s * self.embedding_model.sample_rate)
    # Librosa frames as [frame_length, batch], so need a transpose.
    framed_audio = librosa.util.frame(audio, frame_length, hop_length).T
    return framed_audio

  def serialize_tensor(self, tensor: tf.Tensor) -> np.ndarray:
    serialized = tf.io.serialize_tensor(tensor)
    return serialized.numpy()

  def embed(
      self, file_id: str, audio: np.ndarray, timestamp_offset: int
  ) -> tf.train.Example:
    """Apply the embedding model to the target audio array."""
    logging.info('...creating embeddings (%s)', file_id)
    outputs = self.embedding_model.embed(audio)
    feature = {
        FILE_NAME: bytes_feature(bytes(file_id, encoding='utf8')),
        TIMESTAMP: int_feature(timestamp_offset),
    }
    if self.write_embeddings and outputs.embeddings is not None:
      feature[EMBEDDING] = bytes_feature(
          self.serialize_tensor(outputs.embeddings)
      )
      feature[EMBEDDING_SHAPE] = (int_feature(outputs.embeddings.shape),)
    if self.write_logits and outputs.logits is not None:
      for logits_key, value in outputs.logits.items():
        feature[logits_key] = bytes_feature(self.serialize_tensor(value))
        feature[logits_key + '_shape'] = int_feature(value.shape)
    if self.write_separated_audio and outputs.separated_audio is not None:
      feature[SEPARATED_AUDIO] = bytes_feature(
          self.serialize_tensor(outputs.separated_audio)
      )
      feature[SEPARATED_AUDIO_SHAPE] = int_feature(
          outputs.separated_audio.shape
      )
    if self.write_raw_audio:
      feature[RAW_AUDIO] = bytes_feature(
          self.serialize_tensor(tf.constant(audio, dtype=tf.float32))
      )
      feature[RAW_AUDIO_SHAPE] = int_feature(audio.shape)
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    beam.metrics.Metrics.counter('beaminference', 'examples_processed').inc()
    return ex

  @beam.typehints.with_output_types(Any)
  def process(self, source_info: SourceInfo, crop_s: float = -1.0):
    """Process a source.

    Args:
      source_info: SourceInfo describing the audio to process.
      crop_s: If >0, only the first crop_s seconds will be used. Helpful for
        dry-run testing.

    Returns:
      A TFExample.
    """
    file_name = os.path.basename(source_info.filepath)

    logging.info('...loading audio (%s)', source_info.filepath)
    audio = self.load_audio(source_info.filepath)
    if audio is None:
      beam.metrics.Metrics.counter('beaminference', 'load_audio_error').inc()
      logging.error('Failed to load audio : %s', source_info.filepath)
      return

    if audio.shape[0] < MIN_AUDIO_S * self.embedding_model.sample_rate:
      beam.metrics.Metrics.counter('beaminference', 'short_audio_error').inc()
      logging.error('short audio file : %s', source_info.filepath)
      return

    if source_info.num_shards > 1:
      shard_len = audio.shape[0] // source_info.num_shards
      timestamp_offset = source_info.shard_num * shard_len
      audio = audio[timestamp_offset : timestamp_offset + shard_len]
    else:
      timestamp_offset = 0

    if crop_s > 0:
      audio = audio[: int(crop_s * self.embedding_model.sample_rate)]
    elif self.crop_s > 0:
      audio = audio[: int(self.crop_s * self.embedding_model.sample_rate)]
    return [self.embed(file_name, audio, timestamp_offset)]


def build_run_pipeline(base_pipeline, output_dir, source_infos, embed_fn):
  """Create and run a beam pipeline."""
  _ = (
      base_pipeline
      | beam.Create(source_infos)
      | beam.ParDo(embed_fn)
      # When a file is corrupted and can't be loaded EmbedFn
      # returns None. In this case the lambda below returns false, which then
      # filters it out.
      | beam.Filter(lambda x: x)
      | beam.Reshuffle()
      | beam.io.tfrecordio.WriteToTFRecord(
          output_dir,
          coder=beam.coders.ProtoCoder(tf.train.Example),
      )
  )
  metrics = base_pipeline.run().metrics()
  return metrics


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
