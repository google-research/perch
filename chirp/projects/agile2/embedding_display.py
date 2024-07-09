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

"""Object wrapping an embedded audio example for display."""

import dataclasses
import functools
from typing import Callable, Iterator, Sequence

from chirp import audio_utils
from chirp.models import frontend
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import search_results
import IPython
from IPython.display import clear_output
from IPython.display import display as ipy_display
import ipywidgets
from librosa import display as librosa_display
from matplotlib import pyplot as plt
import numpy as np


@dataclasses.dataclass
class QueryDisplay:
  """Object wrapping a query for display."""

  uri: str
  window_size_s: float
  audio: np.ndarray | None = None
  full_spectrogram: np.ndarray | None = None
  window_spectrogram: np.ndarray | None = None
  offset_s: float = 0.0
  sample_rate_hz: int = 32000
  frame_rate: int = 100

  def update_audio(self):
    """Get the audio for this query."""
    if self.audio is None:
      self.audio = np.array(
          audio_utils.load_audio(self.uri, self.sample_rate_hz)
      )

  def update_spectrogram(self):
    """Update the spectrogram for this query."""
    if self.full_spectrogram is None:
      self.full_spectrogram = get_melspec_layer(self.sample_rate_hz)(self.audio)
    start = int(self.offset_s * self.sample_rate_hz)
    end = int(start + self.window_size_s * self.sample_rate_hz)
    self.window_spectrogram = get_melspec_layer(self.sample_rate_hz)(
        self.audio[start:end]
    )

  def get_audio_window(self) -> np.ndarray | None:
    """Get the audio window for this query."""
    if self.audio is None:
      self.update_audio()
    if self.audio is None:
      return None
    start = int(self.offset_s * self.sample_rate_hz)
    end = int(start + self.window_size_s * self.sample_rate_hz)
    return self.audio[start:end]

  def plot_spectrogram(self, spec: np.ndarray):
    fig, ax = plt.subplots()
    img = librosa_display.specshow(
        spec,
        sr=self.sample_rate_hz,
        y_axis='mel',
        x_axis='time',
        hop_length=self.sample_rate_hz // self.frame_rate,
        cmap='Greys',
        ax=ax,
    )
    return fig, img, ax

  def display(self):
    """Display the audio, spectrogram, and label buttons."""
    self.update_audio()
    self.update_spectrogram()
    # Display full-audio spectrogram
    if self.full_spectrogram is not None:
      librosa_display.specshow(
          self.full_spectrogram.T,
          sr=self.sample_rate_hz,
          y_axis='mel',
          x_axis='time',
          hop_length=self.sample_rate_hz // self.frame_rate,
          cmap='Greys',
      )
      window_end = self.offset_s + self.window_size_s
      plt.plot([self.offset_s, self.offset_s], [0, 16000], 'r:')
      plt.plot([window_end, window_end], [0, 16000], 'r:')
      plt.show()

    # Display the current window.
    if self.window_spectrogram is not None:
      librosa_display.specshow(
          self.window_spectrogram.T,
          sr=self.sample_rate_hz,
          y_axis='mel',
          x_axis='time',
          hop_length=self.sample_rate_hz // self.frame_rate,
          cmap='Greys',
      )
      plt.show()

    ipy_display(IPython.display.Audio(self.audio, rate=self.sample_rate_hz))
    window_st = int(self.offset_s * self.sample_rate_hz)
    window_end = int(window_st + self.window_size_s * self.sample_rate_hz)
    audio_window = self.audio[window_st:window_end]
    ipy_display(IPython.display.Audio(audio_window, rate=self.sample_rate_hz))

  def display_interactive(self):
    """Create an interactive slider for the offset."""
    self.update_audio()
    if self.audio is None:
      print(f'No audio loaded for {self.uri}.')
      return
    self.update_spectrogram()
    slider = ipywidgets.FloatSlider(
        value=self.offset_s,
        min=0.0,
        max=self.audio.shape[0] / self.sample_rate_hz,
        continuous_update=False,
        description='offset_s',
    )

    def update(x):
      self.offset_s = x
      clear_output(wait=True)
      self.display()

    ipywidgets.interact(update, x=slider)


@dataclasses.dataclass
class EmbeddingDisplay:
  """Object wrapping an embedded audio example for display."""

  embedding_id: int
  dataset_name: str
  uri: str
  offset_s: float
  score: float
  widgets: dict[str, ipywidgets.Button] = dataclasses.field(
      default_factory=dict
  )
  sample_rate_hz: int = 32000
  frame_rate: int = 100
  audio: np.ndarray | None = None
  spectrogram: np.ndarray | None = None

  def _make_label_button(self, button_label: str) -> ipywidgets.Button:
    """Create an ipywidget button for the given label."""

    def button_callback(x):
      if x.value == 0:
        x.value = 1
        x.button_style = 'success'
      elif x.value == 1:
        x.button_style = 'warning'
        x.value = -1
      elif x.value == -1:
        x.button_style = ''
        x.value = 0
      else:
        raise ValueError(f'Unexpected button value: {x.value}')

    button = ipywidgets.Button(
        description=button_label,
        disabled=False,
        button_style='',
    )
    button.value = 0
    button.on_click(button_callback)
    return button

  def _make_label_widgets(self, labels: Sequence[str]):
    """Create ipywidget buttons for the given labels."""
    # Create widgets for the labels as needed.
    for label in labels:
      if label not in self.widgets:
        self.widgets[label] = self._make_label_button(label)

  def display(
      self,
      labels: Sequence[str] = (),
      rank: int = -1,
      show_score: bool = True,
      button_columns: int = 4,
  ):
    """Display the audio, spectrogram, and label buttons."""
    self._make_label_widgets(labels)
    if self.audio is None:
      print(f'No audio loaded for {self.uri} :: {self.offset_s}.')
      return
    if self.spectrogram is None:
      print(f'No spectrogram computed for {self.uri} :: {self.offset_s}.')
      return
    # Display spectrogram
    librosa_display.specshow(
        self.spectrogram.T,
        sr=self.sample_rate_hz,
        y_axis='mel',
        x_axis='time',
        hop_length=self.sample_rate_hz // self.frame_rate,
        cmap='Greys',
    )
    plt.show()
    # Display audio
    ipy_display(IPython.display.Audio(self.audio, rate=self.sample_rate_hz))
    print(f'dataset name : {self.dataset_name}')
    print(f'source uri   : {self.uri}')
    print(f'offset_s     : {self.offset_s:.2f}')
    if rank > 0:
      print(f'rank         : {rank}')
    if show_score:
      print(f'score        : {(self.score):.2f}')

    # Display widgets
    grid = ipywidgets.GridspecLayout(
        n_rows=len(self.widgets) // button_columns + 1,
        n_columns=button_columns,
    )
    for i, button in enumerate(self.widgets.values()):
      if button is None:
        continue
      row = i // button_columns
      col = i % button_columns
      grid[row, col] = button
    ipy_display(grid)

  def harvest_labels(self, provenance: str) -> Sequence[interface.Label]:
    """Get the labels for this example."""
    labels = []
    for lbl, w in self.widgets.items():
      if not w.value:
        continue
      elif w.value == -1:
        lbl_type = interface.LabelType.NEGATIVE
      elif w.value == 1:
        lbl_type = interface.LabelType.POSITIVE
      else:
        raise ValueError(f'Unexpected button value: {w.value}')
      labels.append(
          interface.Label(
              embedding_id=self.embedding_id,
              label=lbl,
              type=lbl_type,
              provenance=provenance,
          )
      )
    return labels


@dataclasses.dataclass
class EmbeddingDisplayGroup:
  """Group of EmbeddingDisplay objects."""

  members: Sequence[EmbeddingDisplay]
  melspec_layer: Callable[[np.ndarray], np.ndarray]
  audio_loader: Callable[[str, float], np.ndarray]
  current_page: int = 0
  results_per_page: int = 10
  sample_rate_hz: int = 32000

  @classmethod
  def create(
      cls, members: Sequence[EmbeddingDisplay], sample_rate_hz: int, **kwargs
  ) -> 'EmbeddingDisplayGroup':
    """Create an EmbeddingDisplayGroup from a list of EmbeddingDisplay objects."""
    melspec_layer = get_melspec_layer(sample_rate_hz)
    return cls(
        members=members,
        melspec_layer=melspec_layer,
        sample_rate_hz=sample_rate_hz,
        **kwargs,
    )

  @classmethod
  def from_search_results(
      cls,
      results: search_results.TopKSearchResults,
      db: interface.GraphSearchDBInterface,
      sample_rate_hz: int,
      frame_rate: int,
      **kwargs,
  ) -> 'EmbeddingDisplayGroup':
    """Create an EmbeddingDisplayGroup from a Hoplite TopKSearchResults object."""
    members = []
    for result in results:
      source = db.get_embedding_source(result.embedding_id)
      members.append(
          EmbeddingDisplay(
              embedding_id=result.embedding_id,
              dataset_name=source.dataset_name,
              uri=source.source_id,
              offset_s=source.offsets[0],
              score=result.sort_score,
              sample_rate_hz=sample_rate_hz,
              frame_rate=frame_rate,
          )
      )
    return cls.create(members=members, sample_rate_hz=sample_rate_hz, **kwargs)

  def iterator_with_audio(
      self,
      current_page_only: bool = False,
  ) -> Iterator[EmbeddingDisplay]:
    """Fetch audio for members of the group and compute spectrograms."""
    if len(self.members) > self.results_per_page and current_page_only:
      min_idx = self.current_page * self.results_per_page
      max_idx = min(
          (self.current_page + 1) * self.results_per_page, len(self.members)
      )
      targets = self.members[min_idx:max_idx]
    else:
      targets = self.members

    # We want to iterate over all members, but only load audio for those that
    # don't already have it.
    needs_audio = [m.audio is None for m in targets]
    needs_audio_targets = [m for m in targets if m.audio is None]

    filepaths = [m.uri for m in needs_audio_targets]
    offsets = [m.offset_s for m in needs_audio_targets]
    audio_iter_ = audio_utils.multi_load_audio_window(
        filepaths, offsets, self.audio_loader
    )
    for member, is_dispatched in zip(targets, needs_audio):
      if is_dispatched:
        got_audio = next(audio_iter_)
        if got_audio is not None:
          member.audio = got_audio
          member.spectrogram = self.melspec_layer(got_audio)
      yield member

  @property
  def num_pages(self) -> int:
    return len(self.members) // self.results_per_page

  def increment_page(self, inc: int):
    self.current_page += inc
    self.current_page = min(self.current_page, self.num_pages)
    self.current_page = max(0, self.current_page)

  def display(
      self,
      positive_labels: Sequence[str] = (),
      show_score: bool = True,
      paged_mode: bool = True,
  ):
    """Display the audio, spectrogram, and label buttons."""
    clear_output()
    if paged_mode:
      rank_offset = self.current_page * self.results_per_page
      print(f'Page {self.current_page + 1} of {self.num_pages + 1}')
      print('-' * 80 + '\n')
    else:
      rank_offset = 0

    # Attach audio as needed.
    member_iterator = self.iterator_with_audio(current_page_only=paged_mode)

    # Display the selected members.
    for r, member in enumerate(member_iterator):
      member.display(
          positive_labels,
          show_score=show_score,
          rank=r + rank_offset,
          button_columns=1,
      )
      print('\n' + '-' * 80)

    # Display next/prev page buttons as needed.
    if paged_mode:
      print(f'Page {self.current_page + 1} of {self.num_pages + 1}')

      def increment_page_callback(unused_x, inc):
        self.increment_page(inc)
        self.display(
            positive_labels, show_score=show_score, paged_mode=paged_mode
        )

      next_page_button = ipywidgets.Button(
          description='Next Page', disabled=False
      )
      next_page_button.on_click(lambda x: increment_page_callback(x, 1))
      prev_page_button = ipywidgets.Button(
          description='Prev Page', disabled=False
      )
      prev_page_button.on_click(lambda x: increment_page_callback(x, -1))
      ipy_display(prev_page_button)
      ipy_display(next_page_button)

  def harvest_labels(self, provenance) -> Sequence[interface.Label]:
    labels = []
    for member in self.members:
      labels.extend(member.harvest_labels(provenance))
    return labels


@functools.cache
def get_melspec_layer(
    sample_rate_hz: int, root=4.0
) -> Callable[[np.ndarray], np.ndarray]:
  """Creates a melspec layer for easy visualization."""
  # Usage: melspec_layer.apply({}, audio)
  stride = sample_rate_hz // 100
  melspec_layer = frontend.MelSpectrogram(
      96,
      stride,
      4 * stride,
      sample_rate_hz,
      (60.0, sample_rate_hz / 2.0),
      scaling_config=frontend.PCENScalingConfig(root=root, bias=0.0),
  )
  return lambda x: np.array(melspec_layer.apply({}, x[np.newaxis, :])[0])
