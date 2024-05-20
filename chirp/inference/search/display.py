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

"""Utility functions for displaying audio and results in Colab/Jupyter."""

import dataclasses
import functools
from typing import Sequence

from chirp.inference.search import bootstrap
from chirp.inference.search import search
from chirp.models import frontend
import IPython
from IPython.display import clear_output
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
  melspec_layer = frontend.MelSpectrogram(
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
  if audio.shape[0] < sample_rate / 100 + 1:
    # Center pad if audio is too short.
    zs = np.zeros([sample_rate // 10], dtype=audio.dtype)
    audio = np.concatenate([zs, audio, zs], axis=0)
  melspec = melspec_layer.apply({}, audio[np.newaxis, :])[0]
  plot_melspec(melspec, newfig=newfig, sample_rate=sample_rate, frame_rate=100)
  plt.show()
  if display_audio:
    ipy_display(IPython.display.Audio(audio, rate=sample_rate))


def _make_result_buttons(button_labels: Sequence[str]):
  """Creates buttons for selected labels."""

  def button_callback(x):
    x.value = not x.value
    if x.value:
      x.button_style = 'success'
    else:
      x.button_style = ''

  buttons = []
  for lbl in button_labels:
    check = ipywidgets.Button(
        description=lbl,
        disabled=False,
        button_style='',
    )
    check.value = False
    check.on_click(button_callback)

    buttons.append(check)
  return buttons


def _make_result_radio_buttons(button_labels: Sequence[str]):
  """Make radio buttons with the indicated labels."""
  b = ipywidgets.RadioButtons(options=button_labels)
  # Explicitly set value to None to avoid pre-selecting the first option.
  b.value = None
  return [b]


def display_search_results(
    project_state: bootstrap.BootstrapState,
    results: search.TopKSearchResults,
    embedding_sample_rate: int,
    checkbox_labels: Sequence[str] = (),
    exclusive_labels=False,
    rank_offset: int = 0,
    **kwargs,
):
  """Display search results, and add audio and annotation info to results."""
  results_iterator = project_state.search_results_audio_iterator(
      results, **kwargs
  )

  # Parallel load the audio windows.
  for rank, result in enumerate(results_iterator):
    if result.audio is not None:
      plot_audio_melspec(result.audio, embedding_sample_rate)
      plt.show()
    else:
      print('Failed to load audio for result.')
    print(f'rank        : {rank + rank_offset}')
    print(f'source file : {result.filename}')
    offset_s = result.timestamp_offset
    print(f'offset_s    : {offset_s:.2f}')
    print(f'score       : {(result.score):.2f}')

    if not result.label_widgets:
      if exclusive_labels:
        result.label_widgets = _make_result_radio_buttons(checkbox_labels)
      else:
        result.label_widgets = _make_result_buttons(checkbox_labels)

    for b in result.label_widgets:
      ipy_display(b)

    print('-' * 80)


@dataclasses.dataclass
class PageState:
  max_page: int
  curr_page: int = 0

  def increment(self, inc):
    self.curr_page += inc
    self.curr_page = min(self.max_page, self.curr_page)
    self.curr_page = max(0, self.curr_page)


def display_paged_results(
    all_results: search.TopKSearchResults,
    page_state: PageState,
    samples_per_page: int = 10,
    **kwargs,
):
  """Display search results in pages."""

  def increment_page_callback(unused_x, inc, page_state):
    page_state.increment(inc)
    display_page(page_state)

  next_page_button = ipywidgets.Button(description='Next Page', disabled=False)
  next_page_button.on_click(lambda x: increment_page_callback(x, 1, page_state))
  prev_page_button = ipywidgets.Button(description='Prev Page', disabled=False)
  prev_page_button.on_click(
      lambda x: increment_page_callback(x, -1, page_state)
  )

  def display_page(page_state):
    clear_output()
    num_pages = len(all_results.search_results) // samples_per_page
    page = page_state.curr_page
    print(f'Results Page: {page} / {num_pages}')
    st, end = page * samples_per_page, (page + 1) * samples_per_page
    results_page = search.TopKSearchResults(
        top_k=samples_per_page,
        search_results=all_results.search_results[st:end],
    )
    display_search_results(
        results=results_page, rank_offset=page * samples_per_page, **kwargs
    )
    print(f'Results Page: {page} / {num_pages}')
    ipy_display(prev_page_button)
    ipy_display(next_page_button)

  # Display the first page.
  display_page(page_state)
