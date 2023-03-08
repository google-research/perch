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

"""Utilities for manipulating TF Examples."""

from typing import Sequence

from chirp.inference import interface
import numpy as np
import tensorflow as tf


# Feature keys.
FILE_NAME = 'filename'
TIMESTAMP = 'timestamp_offset'
EMBEDDING = 'embedding'
EMBEDDING_SHAPE = 'embedding_shape'
LOGITS = 'logits'
SEPARATED_AUDIO = 'separated_audio'
SEPARATED_AUDIO_SHAPE = 'separated_audio_shape'
RAW_AUDIO = 'raw_audio'
RAW_AUDIO_SHAPE = 'raw_audio_shape'


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


def get_example_parser(logit_names: Sequence[str] | None = None):
  """Create a parser for decoding inference library TFExamples."""
  features = get_feature_description(logit_names=logit_names)

  def _parser(ex):
    ex = tf.io.parse_single_example(ex, features)
    if ex[EMBEDDING]:
      ex[EMBEDDING] = tf.io.parse_tensor(ex[EMBEDDING], tf.float32)
    if ex[SEPARATED_AUDIO]:
      ex[SEPARATED_AUDIO] = tf.io.parse_tensor(ex[SEPARATED_AUDIO], tf.float32)
    for logit_key in logit_names:
      if ex[logit_key]:
        ex[logit_key] = tf.io.parse_tensor(ex[logit_key], tf.float32)
    if ex[RAW_AUDIO]:
      ex[RAW_AUDIO] = tf.io.parse_tensor(ex[RAW_AUDIO], tf.float32)
    return ex

  return _parser


def serialize_tensor(tensor: tf.Tensor) -> np.ndarray:
  serialized = tf.io.serialize_tensor(tensor)
  return serialized.numpy()


def model_outputs_to_tf_example(
    model_outputs: interface.InferenceOutputs,
    file_id: str,
    audio: np.ndarray,
    timestamp_offset: int,
    write_embeddings: bool,
    write_logits: bool,
    write_separated_audio: bool,
    write_raw_audio: bool,
) -> tf.train.Example:
  """Create a TFExample from InferenceOutputs."""
  feature = {
      FILE_NAME: bytes_feature(bytes(file_id, encoding='utf8')),
      TIMESTAMP: int_feature(timestamp_offset),
  }
  if write_embeddings and model_outputs.embeddings is not None:
    feature[EMBEDDING] = bytes_feature(
        serialize_tensor(model_outputs.embeddings)
    )
    feature[EMBEDDING_SHAPE] = (int_feature(model_outputs.embeddings.shape),)
  if write_logits and model_outputs.logits is not None:
    for logits_key, value in model_outputs.logits.items():
      feature[logits_key] = bytes_feature(serialize_tensor(value))
      feature[logits_key + '_shape'] = int_feature(value.shape)
  if write_separated_audio and model_outputs.separated_audio is not None:
    feature[SEPARATED_AUDIO] = bytes_feature(
        serialize_tensor(model_outputs.separated_audio)
    )
    feature[SEPARATED_AUDIO_SHAPE] = int_feature(
        model_outputs.separated_audio.shape
    )
  if write_raw_audio:
    feature[RAW_AUDIO] = bytes_feature(
        serialize_tensor(tf.constant(audio, dtype=tf.float32))
    )
    feature[RAW_AUDIO_SHAPE] = int_feature(audio.shape)
  ex = tf.train.Example(features=tf.train.Features(feature=feature))
  return ex


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
