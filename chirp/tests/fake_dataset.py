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

"""A fake dataset for testing."""

from typing import Tuple

from chirp.data import bird_taxonomy
import numpy as np


def window_signal(signal, win_len=2000):
  ramp = np.linspace(0.0, 1.0, win_len // 2)
  signal[: win_len // 2] *= ramp
  signal[-win_len // 2 :] *= ramp[::-1]
  return signal


def make_dsp_chirp(n_samples, start_f, stop_f, linear=True):
  """Creates a np.array with a sine sweep, aka a `chirp`."""
  time = np.linspace(0.0, 1.0, n_samples)
  space_fn = np.linspace if linear else np.geomspace
  sweep_f = space_fn(start_f, stop_f, n_samples)
  chirp_ = np.sin(np.pi * sweep_f * time)
  return window_signal(chirp_)


def make_random_dsp_chirps(total_len, gain=0.25):
  """Makes a clip with two random sine sweeps (chirps)."""
  n_samples = total_len // 4
  chirp_clip = np.zeros((total_len,))

  ch1_beg, ch2_beg = np.random.choice(np.arange(300, 2001, 50), size=(2,))
  ch1_end, ch2_end = np.random.choice(np.arange(800, 3801, 50), size=(2,))
  chirp1 = make_dsp_chirp(n_samples, ch1_beg, ch1_end, linear=False)
  chirp2 = make_dsp_chirp(n_samples, ch2_beg, ch2_end, linear=True)
  chirp_clip[n_samples : n_samples * 2] = chirp1
  chirp_clip[int(n_samples * 2.5) : int(n_samples * 3.5)] = chirp2
  return chirp_clip * gain


class FakeDataset(bird_taxonomy.BirdTaxonomy):
  """Fake dataset."""

  def _split_generators(self, dl_manager):
    return {
        'train': self._generate_examples(100),
        'test': self._generate_examples(20),
    }

  @staticmethod
  def _make_signal(shape: Tuple[int, ...]) -> np.ndarray:
    return np.random.uniform(-1.0, 0.99, shape)

  def _generate_one_example(self, i):
    return {
        'audio': self._make_signal(self.info.features['audio'].shape).astype(
            np.float32
        ),
        'recording_id': i,
        'segment_id': -1 + i,
        'segment_start': 17,
        'segment_end': 17 + self.info.features['audio'].shape[0],
        'label': np.random.choice(self.info.features['label'].names, size=[3]),
        'bg_labels': np.random.choice(
            self.info.features['bg_labels'].names, size=[2]
        ),
        'filename': 'placeholder',
        'quality_score': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F']),
        'license': '//creativecommons.org/licenses/by-nc-sa/4.0/',
        'country': np.random.choice(['South Africa', 'Colombia', 'Namibia']),
        'altitude': str(np.random.uniform(0, 3000)),
        'length': np.random.choice(['1:10', '0:01']),
        'bird_seen': np.random.choice(['yes', 'no']),
        'latitude': str(np.random.uniform(0, 90)),
        'longitude': str(np.random.uniform(0, 90)),
        'playback_used': 'yes',
        'recordist': 'N/A',
        'remarks': 'N/A',
        'sound_type': np.random.choice(['call', 'song']),
    }

  def _generate_examples(self, num_examples):
    for i in range(num_examples):
      yield i, self._generate_one_example(i)


class FakeChirpDataset(FakeDataset):
  """Fake dataset with DSP chirps; useful for debugging separation."""

  @staticmethod
  def _make_signal(shape: Tuple[int, ...]) -> np.ndarray:
    return make_random_dsp_chirps(shape[0])
