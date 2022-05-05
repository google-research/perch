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

"""Tools for colab data handling and analysis."""

import os
import tempfile
from typing import Optional
import google3
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

from google3.audio.hearing.daredevil.tools.ffmpeg_util import TranscodeAudio
from google3.pyglib import gfile


def prstats(title: str, ar: jnp.ndarray):
  """Print summary statistics for an array."""
  tmpl = ('% 16s : \t'
          'shape: % 16s\t'
          'min: %6.2f\t'
          'mean: %6.2f\t'
          'max: %6.2f\t'
          'std: %6.2f')
  print(tmpl %
        (title, np.shape(ar), np.min(ar), np.mean(ar), np.max(ar), np.std(ar)))


def progress_dot(i: int, print_mod: int = 10, break_mod: Optional[int] = None):
  """Print a dot, with occasional line breaks."""
  if break_mod is None:
    break_mod = 25 * print_mod
  if (i + 1) % break_mod == 0:
    print('.')
  elif (i + 1) % print_mod == 0 or print_mod <= 1:
    print('.', end='')


def plot_melspec(melspec: jnp.ndarray, newfig: bool = False, lbl: str = ''):
  if newfig:
    plt.figure(figsize=(12, 5))
  plt.grid('off')
  pic = np.abs(np.flipud(melspec.T))
  plt.imshow(pic, interpolation='none', aspect='auto', cmap='Greys')
  if lbl:
    plt.xlabel(lbl)


def load_audio(filepath: str, sample_rate: int, bitdepth=16):
  """LoadAudio loads an audio file.

  Resamples to sample_rate, drops all but the 0th channel.

  Args:
    filepath: Audio file to load.
    sample_rate: Target sample rate. Set to 0 to avoid resampling.
    bitdepth: Scaling term. Set to 0 to avoid rescaling.

  Returns:
    sample_rate, numpy array of samples.
  """
  tmpdir = tempfile.gettempdir()
  with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
    gfile.Copy(filepath, tmp.name, overwrite=True)
  uuid = tmp.name.split('/')[-1]
  output_filepath = os.path.join(tmpdir, 'out_' + uuid)
  (wav, _) = TranscodeAudio(
      tmp.name, output_filepath, output_sample_rate=sample_rate)
  sr, array = scipy.io.wavfile.read(wav, mmap=True)
  if len(array.shape) > 1:
    array = array[:, 0]
  if bitdepth > 0:
    array = 1.0 * array / 2**bitdepth
  if sample_rate > 0 and sr != sample_rate:
    target_samples = int(sample_rate / sr * array.shape[0])
    array = scipy.signal.resample(array, target_samples)
  try:
    os.remove(tmp.name)
    os.remove(output_filepath)
  except FileNotFoundError:
    pass
  return sample_rate, array


def parallel_load_wavs(filepattern: str, sample_rate: int):
  """Load a collection of wav files using tf.dataset parallelism."""
  files_ds = tf.data.Dataset.from_tensor_slices(gfile.Glob(filepattern))

  def _readwav(filename):
    data = tf.io.read_file(filename)
    wav, sr = tf.audio.decode_wav(contents=data)
    return filename, wav, sr

  ds = files_ds.map(_readwav, num_parallel_call=20)
  ds = ds.prefetch(50)
  audios = {}
  for filename, audio, sr in ds:
    filename = filename.numpy().decode('utf-8')
    audio = audio.numpy()
    if sample_rate > 0 and sr != sample_rate:
      target_samples = int(sample_rate / sr * audio.shape[0])
      audio = scipy.signal.resample(audio, target_samples)
    audios[filename] = audio
  return audios
