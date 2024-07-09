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

"""Utilities for testing agile modeling functionality."""

import os

import numpy as np
from scipy.io import wavfile


def make_wav_files(
    base_path, classes, filenames, file_len_s=1.0, sample_rate_hz=16000
):
  """Create a pile of wav files in a directory structure."""
  rng = np.random.default_rng(seed=42)
  for subdir in classes:
    subdir_path = os.path.join(base_path, subdir)
    os.mkdir(subdir_path)
    for filename in filenames:
      with open(
          os.path.join(subdir_path, f'{filename}_{subdir}.wav'), 'wb'
      ) as f:
        noise = rng.normal(scale=0.2, size=int(file_len_s * sample_rate_hz))
        wavfile.write(f, sample_rate_hz, noise)
  audio_glob = os.path.join(base_path, '*/*.wav')
  return audio_glob
