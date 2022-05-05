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

"""Tests for audio utilities."""
from chirp import audio_utils
from jax import numpy as jnp
from jax import random
from librosa.core import spectrum
import numpy as np


def test_compute_melspec():
  sample_rate_hz = 22050
  audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, sample_rate_hz))
  noise = 0.01 * random.normal(random.PRNGKey(0), (2, 3, sample_rate_hz))

  kwargs = {
      "sample_rate_hz": 22050,
      "melspec_depth": 160,
      "melspec_frequency": 100,
      "frame_length_secs": 0.08,
      "upper_edge_hz": 8000.0,
      "lower_edge_hz": 60.0
  }

  # Test mel-spectrogram shape and batching
  melspec = audio_utils.compute_melspec(
      audio + noise[0, 0], scaling_config=None, **kwargs)

  assert melspec.shape == (100, 160)

  batch_melspec = audio_utils.compute_melspec(
      audio + noise, scaling_config=None, **kwargs)

  assert batch_melspec.shape == (2, 3, 100, 160)
  np.testing.assert_allclose(batch_melspec[0, 0], melspec)

  # Test normalization
  melspec = audio_utils.compute_melspec(
      audio + noise[0, 0],
      scaling_config=audio_utils.LogScalingConfig(),
      **kwargs)
  assert melspec.shape == (100, 160)

  melspec = audio_utils.compute_melspec(
      audio + noise[0, 0],
      scaling_config=audio_utils.PCENScalingConfig(),
      **kwargs)
  assert melspec.shape == (100, 160)


def test_fixed_pcen():
  sample_rate_hz = 22050
  audio = jnp.sin(jnp.linspace(0.0, 440 * jnp.pi, sample_rate_hz))
  noise = 0.01 * random.normal(random.PRNGKey(0), (
      1,
      sample_rate_hz,
  ))
  filterbank_energy = audio_utils.compute_melspec(
      audio + noise[0, 0],
      sample_rate_hz=sample_rate_hz,
      melspec_depth=160,
      melspec_frequency=100,
      frame_length_secs=0.08,
      upper_edge_hz=8000.0,
      lower_edge_hz=60.0,
      scaling_config=None)

  gain = 0.5
  smoothing_coef = 0.1
  bias = 2.0
  root = 2.0
  eps = 1e-6

  out = audio_utils.fixed_pcen(
      filterbank_energy,
      gain=gain,
      smoothing_coef=smoothing_coef,
      bias=bias,
      root=root,
      eps=eps)[0]
  librosa_out = spectrum.pcen(
      filterbank_energy,
      b=smoothing_coef,
      gain=gain,
      bias=bias,
      power=1 / root,
      eps=eps,
      # librosa starts with an initial state of (1 - s), we start with x[0]
      zi=[(1 - smoothing_coef) * filterbank_energy[..., 0, :]],
      axis=-2)

  np.testing.assert_allclose(out, librosa_out, rtol=1e-2)
