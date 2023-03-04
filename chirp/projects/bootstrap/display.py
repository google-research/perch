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

"""Utility functions for displaying audio and results in Colab/Jupyter."""

import functools
import time
from typing import Sequence


from chirp.models import frontend
from chirp.projects.bootstrap import search
import IPython
from IPython.display import display as ipy_display
import ipywidgets
from librosa import display as librosa_display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@functools.cache
def get_melspec_layer(sample_rate: int):
  """Creates a melspec layer for easy visualization."""
  # Usage: melspec_layer.apply({}, audio)
  stride = sample_rate // 100
  melspec_layer = frontend.MelSpectrogram(
      160,
      stride,
      int(0.08 * sample_rate),
      sample_rate,
      (60.0, sample_rate / 2.0),
      scaling_config=frontend.PCENScalingConfig(root=8.0, bias=0.0),
  )
  return melspec_layer


def plot_melspec(
    melspec: np.ndarray,
    newfig: bool = False,
    sample_rate: int = 32000,
    frame_rate: int = 100,
    **specshow_kwargs,
):
  """Plot a melspectrogram."""
  if newfig:
    plt.figure(figsize=(12, 5))
  librosa_display.specshow(
      melspec.T,
      sr=sample_rate,
      y_axis='mel',
      x_axis='time',
      hop_length=sample_rate // frame_rate,
      cmap='Greys',
      **specshow_kwargs,
  )


def plot_audio_melspec(
    audio: np.ndarray,
    sample_rate: int,
    newfig: bool = False,
    display_audio=True,
):
  """Plot a melspectrogram from audio."""
  melspec_layer = get_melspec_layer(sample_rate)
  melspec = melspec_layer.apply({}, audio[np.newaxis, :])[0]
  plot_melspec(melspec, newfig=newfig, sample_rate=sample_rate, frame_rate=100)
  plt.show()
  if display_audio:
    ipy_display(IPython.display.Audio(audio, rate=sample_rate))


def display_search_results(
    results: search.TopKSearchResults,
    embedding_sample_rate: int,
    source_map: dict[str, str],
    window_s: float = 5.0,
    checkbox_labels: Sequence[str] = (),
):
  """Display search results, and add audio and annotation info to results."""

  # TODO(tomdenton): Find ways to load lots of snippets from wavs quickly.
  # We have to read the entire source file to get the 5s chunk we want.
  # This is obviously terribly slow.
  # Speed it up by abusing the TF Dataset to get parallelized file reads.
  def _results_generator():
    for r in results.search_results:
      filepath = source_map[r.filename]
      yield hash(r), filepath

  ds = tf.data.Dataset.from_generator(
      _results_generator, (tf.int64, tf.string), output_shapes=([], [])
  )

  def _parser(result_hash, filepath):
    data = tf.io.read_file(filepath)
    audio, sr = tf.audio.decode_wav(data, 1)
    return result_hash, audio, sr

  results_map = {hash(r): r for r in results.search_results}
  ds = ds.map(_parser, num_parallel_calls=tf.data.AUTOTUNE)
  st = time.time()
  for result_hash, result_audio, audio_sr in ds.as_numpy_iterator():
    r = results_map[result_hash]
    st = int(r.timestamp_offset / embedding_sample_rate * audio_sr)
    end = int(st + window_s * audio_sr)
    result_audio_window = result_audio[st:end, 0]
    plot_audio_melspec(result_audio_window, audio_sr)
    plt.show()
    print(f'source file: {r.filename}')
    offset_s = r.timestamp_offset / audio_sr
    print(f'offset:      {offset_s:6.2f}')
    print(f'distance:    {(r.distance + results.distance_offset):6.2f}')
    label_widgets = []
    for lbl in checkbox_labels:
      check = ipywidgets.Checkbox(description=lbl, value=False)
      label_widgets.append(check)
      ipy_display(check)

    # Attach audio and widgets to the SearchResult.
    r.audio = result_audio_window
    r.label_widgets = label_widgets

    print('-' * 80)
  end = time.time()
  # TODO(tomdenton): Clean up debug messages.
  print(end - st)
