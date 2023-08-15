# coding=utf-8
# Copyright 2023 The Chirp Authors.
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
from typing import Sequence

from chirp import audio_utils
from chirp.models import frontend
from chirp.projects.bootstrap import search
import IPython
from IPython.display import display as ipy_display
import ipywidgets
from librosa import display as librosa_display
import matplotlib.pyplot as plt
import numpy as np


@functools.cache
def get_melspec_layer(sample_rate: int, root=4.0):
  """Creates a melspec layer for easy visualization."""
  # Usage: melspec_layer.apply({}, audio)
  stride = sample_rate // 100
  melspec_layer = frontend.MelSpectrogram(  # pytype: disable=wrong-arg-types  # typed-pandas
      96,
      stride,
      4 * stride,
      sample_rate,
      (60.0, sample_rate / 2.0),
      scaling_config=frontend.PCENScalingConfig(root=root, bias=0.0),
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
    max_workers=5,
):
  """Display search results, and add audio and annotation info to results."""

  # Parallel load the audio windows.
  filepaths = [source_map[r.filename] for r in results]
  offsets = [r.timestamp_offset for r in results]
  for rank, (r, result_audio_window) in enumerate(
      zip(
          results,
          audio_utils.multi_load_audio_window(
              filepaths, offsets, embedding_sample_rate, window_s, max_workers
          ),
      )
  ):
    plot_audio_melspec(result_audio_window, embedding_sample_rate)
    plt.show()
    print(f'rank        : {rank}')
    print(f'source file : {r.filename}')
    offset_s = r.timestamp_offset
    print(f'offset_s    : {offset_s:.2f}')
    print(f'score       : {(r.score):.2f}')
    label_widgets = []

    def button_callback(x):
      x.value = not x.value
      if x.value:
        x.button_style = 'success'
      else:
        x.button_style = ''

    for lbl in checkbox_labels:
      check = ipywidgets.Button(
          description=lbl,
          disabled=False,
          button_style='',
      )
      check.value = False
      check.on_click(button_callback)

      label_widgets.append(check)
      ipy_display(check)
    # Attach audio and widgets to the SearchResult.
    r.audio = result_audio_window
    r.label_widgets = label_widgets

    print('-' * 80)
