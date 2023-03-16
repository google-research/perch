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

import concurrent
import functools
import time
from typing import Sequence

from chirp.models import frontend
from chirp.projects.bootstrap import search
from etils import epath
import IPython
from IPython.display import display as ipy_display
import ipywidgets
import librosa
from librosa import display as librosa_display
import matplotlib.pyplot as plt
import numpy as np
import soundfile


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


def load_audio_window(
    filepath: str, offset_s: float, sample_rate: int, window_size_s: float
):
  """Load an audio window."""
  with epath.Path(filepath).open('rb') as f:
    sf = soundfile.SoundFile(f)
    offset = int(offset_s * sf.samplerate)
    window_size = int(window_size_s * sf.samplerate)
    sf.seek(offset)
    a = sf.read(window_size)
  a = librosa.resample(
      y=a, orig_sr=sf.samplerate, target_sr=sample_rate, res_type='polyphase'
  )
  if len(a.shape) == 2:
    # Downstream ops expect mono audio, so reduce to mono.
    a = a[:, 0]
  return a


def multi_load_audio_window(
    filepaths: Sequence[str],
    offsets: Sequence[int],
    sample_rate: int,
    window_size_s: float,
    max_workers: int = 5,
):
  """Load audio windows in parallel."""
  loader = functools.partial(
      load_audio_window, sample_rate=sample_rate, window_size_s=window_size_s
  )
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    futures = []
    for fp, offset in zip(filepaths, offsets):
      offset_s = offset / sample_rate
      future = executor.submit(loader, offset_s=offset_s, filepath=fp)
      futures.append(future)
    for f in futures:
      yield f.result()


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
  st = time.time()
  filepaths = [source_map[r.filename] for r in results]
  offsets = [r.timestamp_offset for r in results]
  for r, result_audio_window in zip(
      results,
      multi_load_audio_window(
          filepaths, offsets, embedding_sample_rate, window_s, max_workers
      ),
  ):
    plot_audio_melspec(result_audio_window, embedding_sample_rate)
    plt.show()
    print(f'source file: {r.filename}')
    offset_s = r.timestamp_offset / embedding_sample_rate
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
