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

"""Tools for reading/transcoding data from colossus."""

import re
import numpy as np
import scipy
from scipy.io import wavfile
import tensorflow as tf

LATLONG_REGEX = re.compile(r'\((\-?\d+\.\d+?),\s*(\-?\d+\.\d+?)\)')
TIME_OF_DAY_REGEX = re.compile(r'(\d{1,2}:\d\d)')
DATE_REGEX = re.compile(r'(\d\d\d\d-\d{1,2}-\d{1,2})')
ELEV_REGEX = re.compile(r'^(\d+)\s?m')


def LoadAudio(audio_path, target_sr):
  """LoadWavAudio loads a wav file from a path."""

  if '.wav' in audio_path or '.WAV' in audio_path:
    sr, audio = LoadWavAudio(audio_path, sample_rate=target_sr)
    metadata = {}
  else:
    raise Exception('wrong file format, please use .wav')

  if sr != target_sr:
    raise Exception(
        'got wrong sample rate (%s vs %s) from converted file: %s'
        % (sr, target_sr, audio_path)
    )
  return audio, metadata


def CenteredRepeatPad(audio, target_length):
  if audio.shape[0] >= target_length:
    return audio
  padded = audio
  while padded.shape[0] < target_length:
    padded = np.concatenate([audio, padded, audio])
  midpoint = padded.shape[0] // 2
  start = midpoint - target_length // 2
  padded = padded[start : start + target_length]
  return padded


def LoadWavAudio(path, sample_rate, bitdepth=16):
  """LoadWavAudio loads a wav file from a path.

  Resamples to sample_rate, drops all but the 0th channel.

  Args:
    path: Location to load.
    sample_rate: Target sample rate. Set to 0 to avoid resampling.
    bitdepth: Scaling term.

  Returns:
    sample_rate: numpy array of samples.
    array: ?
  """

  sr, array = wavfile.read(path, mmap=True)
  if len(array.shape) > 1:
    array = array[:, 0]
  array = 1.0 * array / 2**bitdepth
  if sample_rate > 0 and sr != sample_rate:
    target_samples = int(sample_rate / sr * array.shape[0])
    array = scipy.signal.resample(array, target_samples)

  return sample_rate, array


def BytesFeature(x, default=''):
  if x is None:
    x = default
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))


def BytesRepFeature(x, default=None):
  if default is None:
    default = []
  if x is None:
    x = default
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))


def FloatFeature(x, default=-1.0):
  if x is None:
    x = default
  return tf.train.Feature(float_list=tf.train.FloatList(value=[x]))


def FloatsFeature(x, default=None):
  if default is None:
    default = []
  if x is None:
    x = default
  return tf.train.Feature(float_list=tf.train.FloatList(value=x))


def IntFeature(x, default=-1):
  if x is None:
    x = default
  if hasattr(x, 'count'):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))
