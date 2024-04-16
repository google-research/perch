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

"""Utilities for manipulating TF Examples."""

import dataclasses
import datetime
import os
from typing import Sequence

from chirp.inference import interface
from etils import epath
import numpy as np
import tensorflow as tf


# Feature keys.
FILE_NAME = 'filename'
TIMESTAMP_S = 'timestamp_s'
EMBEDDING = 'embedding'
EMBEDDING_SHAPE = 'embedding_shape'
LOGITS = 'logits'
SEPARATED_AUDIO = 'separated_audio'
SEPARATED_AUDIO_SHAPE = 'separated_audio_shape'
RAW_AUDIO = 'raw_audio'
RAW_AUDIO_SHAPE = 'raw_audio_shape'
FRONTEND = 'frontend'
FRONTEND_SHAPE = 'frontend_shape'


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
      TIMESTAMP_S: tf.io.FixedLenFeature([], tf.float32),
      EMBEDDING: tf.io.FixedLenFeature([], tf.string, default_value=''),
      EMBEDDING_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      SEPARATED_AUDIO: tf.io.FixedLenFeature([], tf.string, default_value=''),
      SEPARATED_AUDIO_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      FRONTEND: tf.io.FixedLenFeature([], tf.string, default_value=''),
      FRONTEND_SHAPE: tf.io.FixedLenSequenceFeature(
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


def get_example_parser(
    logit_names: Sequence[str] | None = None, tensor_dtype: str = 'float32'
):
  """Create a parser for decoding inference library TFExamples."""
  features = get_feature_description(logit_names=logit_names)

  def _parser(ex):
    ex = tf.io.parse_single_example(ex, features)
    tensor_keys = [EMBEDDING, SEPARATED_AUDIO, RAW_AUDIO, FRONTEND]
    if logit_names is not None:
      tensor_keys.extend(logit_names)
    for key in tensor_keys:
      # Note that we can't use implicit truthiness for string tensors.
      # We are also required to have the same tensor structure and dtype in
      # both conditional branches. So we use an empty tensor when no
      # data is present to parse.
      if ex[key] != tf.constant(b'', dtype=tf.string):
        ex[key] = tf.io.parse_tensor(ex[key], out_type=tensor_dtype)
      else:
        ex[key] = tf.zeros_like([], dtype=tensor_dtype)
    return ex

  return _parser


def create_embeddings_dataset(
    embeddings_dir,
    file_glob: str = '*',
    prefetch: int = 128,
    logit_names: Sequence[str] | None = None,
    tensor_dtype: str = 'float32',
    shuffle_files: bool = False,
):
  """Create a TF Dataset of the embeddings."""
  embeddings_dir = epath.Path(embeddings_dir)
  embeddings_files = [fn.as_posix() for fn in embeddings_dir.glob(file_glob)]
  if shuffle_files:
    np.random.shuffle(embeddings_files)
  ds = tf.data.TFRecordDataset(
      embeddings_files, num_parallel_reads=tf.data.AUTOTUNE
  )

  parser = get_example_parser(
      logit_names=logit_names, tensor_dtype=tensor_dtype
  )
  ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(prefetch)
  return ds


def serialize_tensor(tensor: np.ndarray, tensor_dtype: str) -> np.ndarray:
  tensor = tf.cast(tensor, tensor_dtype)
  serialized = tf.io.serialize_tensor(tensor)
  return serialized.numpy()


def model_outputs_to_tf_example(
    model_outputs: interface.InferenceOutputs,
    file_id: str,
    audio: np.ndarray,
    timestamp_offset_s: float,
    write_embeddings: bool,
    write_logits: bool | Sequence[str],
    write_separated_audio: bool,
    write_raw_audio: bool,
    write_frontend: bool = False,
    tensor_dtype: str = 'float32',
) -> tf.train.Example:
  """Create a TFExample from InferenceOutputs."""
  feature = {
      FILE_NAME: bytes_feature(bytes(file_id, encoding='utf8')),
      TIMESTAMP_S: float_feature(timestamp_offset_s),
  }
  if write_embeddings and model_outputs.embeddings is not None:
    feature[EMBEDDING] = bytes_feature(
        serialize_tensor(model_outputs.embeddings, tensor_dtype)
    )
    feature[EMBEDDING_SHAPE] = (int_feature(model_outputs.embeddings.shape),)

  if write_separated_audio and model_outputs.separated_audio is not None:
    feature[SEPARATED_AUDIO] = bytes_feature(
        serialize_tensor(model_outputs.separated_audio, tensor_dtype)
    )
    feature[SEPARATED_AUDIO_SHAPE] = int_feature(
        model_outputs.separated_audio.shape
    )

  if write_frontend and model_outputs.frontend is not None:
    feature[FRONTEND] = bytes_feature(
        serialize_tensor(model_outputs.frontend, tensor_dtype)
    )
    feature[FRONTEND_SHAPE] = int_feature(model_outputs.frontend.shape)

  if write_raw_audio:
    feature[RAW_AUDIO] = bytes_feature(
        serialize_tensor(tf.constant(audio, dtype=tf.float32), tensor_dtype)
    )
    feature[RAW_AUDIO_SHAPE] = int_feature(audio.shape)

  # Handle writing logits.
  if model_outputs.logits is not None and write_logits:
    logit_keys = tuple(model_outputs.logits.keys())
    if not isinstance(write_logits, bool):
      # Then it's a Sequence[str], so we only keep the relevant keys.
      logit_keys = tuple(k for k in logit_keys if k in write_logits)
    for logits_key in logit_keys:
      logits = model_outputs.logits[logits_key]
      feature[logits_key] = bytes_feature(
          serialize_tensor(logits, tensor_dtype)
      )
      feature[logits_key + '_shape'] = int_feature(logits.shape)

  ex = tf.train.Example(features=tf.train.Features(feature=feature))
  return ex


@dataclasses.dataclass
class EmbeddingsTFRecordMultiWriter:
  """A sharded TFRecord writer."""

  output_dir: str
  filename_pattern: str = 'embeddings-%d-%05d-of-%05d'
  num_files: int = 10
  _writer_index: int = 0

  def write(self, record: str):
    """Write a serialized record."""
    writer = self.writers[self._writer_index]
    writer.write(record)
    self._writer_index = (self._writer_index + 1) % self.num_files

  def flush(self):
    """Flush all files."""
    for writer in self.writers:
      writer.flush()

  def close(self):
    """Close all files."""
    for writer in self.writers:
      writer.close()

  def __enter__(self):
    self.writers = []
    timestamp = int(datetime.datetime.now().timestamp())
    for i in range(self.num_files):
      filepath = os.path.join(
          self.output_dir,
          self.filename_pattern % (timestamp, i, self.num_files),
      )
      self.writers.append(tf.io.TFRecordWriter(filepath))
    return self

  def __exit__(self, *args):
    self.flush()
    self.close()


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


def float_feature(x, default=0.0):
  if x is None:
    x = default
  if hasattr(x, 'count'):
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))
  return tf.train.Feature(float_list=tf.train.FloatList(value=[x]))
