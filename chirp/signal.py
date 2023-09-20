# coding=utf-8
# Copyright 2023 The Perch Authors.
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

"""Signal processing operations.

Ports from `tf.signal`.
"""
import functools

import jax
from jax import lax
from jax import numpy as jnp

_MEL_HIGH_FREQUENCY_Q = 1127.0
_MEL_BREAK_FREQUENCY_HERTZ = 700.0


def hertz_to_mel(frequencies_hertz: jnp.ndarray) -> jnp.ndarray:
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.

  Args:
    frequencies_hertz: An array of frequencies in Hertz.

  Returns:
    An array of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
  """
  return _MEL_HIGH_FREQUENCY_Q * jnp.log1p(
      frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ
  )


def mel_to_hertz(frequencies_mel: jnp.ndarray) -> jnp.ndarray:
  """Converts frequencies in `frequencies_mel` in the mel scale to Hertz.

  Args:
    frequencies_mel: An array of frequencies in the mel scale.

  Returns:
    An array of the same shape and type of `frequencies_mel` containing
    frequencies in Hertz.
  """
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      jnp.expm1(frequencies_mel / _MEL_HIGH_FREQUENCY_Q)
  )


@functools.partial(jax.jit, static_argnums=(0, 1))
def linear_to_mel_weight_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    sample_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> jnp.ndarray:
  """Returns a matrix to warp linear scale spectrograms to the mel scale.

  A port of tf.signal.linear_to_mel_weight_matrix.

  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be `fft_size // 2 + 1`, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    sample_rate: Samples per second of the input signal used to create the
      spectrogram. Used to figure out the frequencies corresponding to each
      spectrogram bin, which dictates how they are mapped into the mel scale.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum. This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.

  Returns:
    An array of shape `(num_spectrogram_bins, num_mel_bins)`.
  """
  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = jnp.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[
      bands_to_zero:
  ]
  spectrogram_bins_mel = hertz_to_mel(linear_frequencies)[:, jnp.newaxis]

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = jnp.linspace(
      hertz_to_mel(lower_edge_hertz),  # pytype: disable=wrong-arg-types  # jax-ndarray
      hertz_to_mel(upper_edge_hertz),  # pytype: disable=wrong-arg-types  # jax-ndarray
      num_mel_bins + 2,
  )

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel = band_edges_mel[jnp.newaxis, :-2]
  center_mel = band_edges_mel[jnp.newaxis, 1:-1]
  upper_edge_mel = band_edges_mel[jnp.newaxis, 2:]

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel
  )
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel
  )

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = jnp.maximum(0.0, jnp.minimum(lower_slopes, upper_slopes))

  # Re-add the zeroed lower bins we sliced out above.
  return jnp.pad(mel_weights_matrix, ((bands_to_zero, 0), (0, 0)))


def frame(
    signal: jnp.ndarray,
    frame_length: int,
    frame_step: int,
    pad_end: bool = False,
    pad_value: float = 0.0,  # pylint: disable=unused-argument
    axis: int = -1,
) -> jnp.ndarray:
  """Split a spectrogram into multiple bands.

  JAX version of `tf.signal.frame`.

  Args:
    signal: A `(..., samples, ...)` array. Rank must be at least 1.
    frame_length: The frame length in samples.
    frame_step: The frame hop size in samples.
    pad_end: Whether to pad the end of `signal` with `pad_value`.
    pad_value: A value to use where the input signal does not exist when
      `pad_end` is True.
    axis: Indicating the axis to frame. Defaults to the last axis. Supports
      negative values for indexing from the end.

  Returns:
    An array of frames, size `(..., num_frames, frame_length, ...)`.
  """
  axis = axis % signal.ndim
  remainder = (signal.shape[axis] - frame_length) % frame_step
  if pad_end and remainder:
    no_pad = ((0, 0),)
    zero_pad = ((0, remainder),)
    pad_width = no_pad * axis + zero_pad + no_pad * (signal.ndim - axis - 1)
    signal = jnp.pad(signal, pad_width, constant_values=pad_value)

  num_frames = (signal.shape[axis] - frame_length) // frame_step + 1
  start_indices = (jnp.arange(num_frames) * frame_step)[:, None]

  # The axis not in offset_dims is where the frames will be put
  offset_dims = tuple(range(axis)) + tuple(range(axis + 1, signal.ndim + 1))

  dimension_numbers = lax.GatherDimensionNumbers(
      offset_dims=offset_dims, collapsed_slice_dims=(), start_index_map=(axis,)
  )
  slice_sizes = signal.shape[:axis] + (frame_length,) + signal.shape[axis + 1 :]
  frames = lax.gather(signal, start_indices, dimension_numbers, slice_sizes)
  return frames
